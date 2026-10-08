"""Bounded HTTP inference with a single model, explicit categories and no saved input text."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..text.schema import CATEGORIES, schema, validate_input

STATIC = Path(__file__).parent / 'static'
MAX_CHARACTERS = 500
MAX_BODY_BYTES = 8192


def create_app(predictor, *, allowed_origins=(), allowed_hosts=('127.0.0.1', 'localhost', '[::1]'), requests_per_minute=30):
    # PSEUDOCODE: warm one predictor -> validate bounded requests -> serialize GPU use -> expose only browser assets.
    if type(requests_per_minute) is not int or requests_per_minute < 1:
        raise ValueError('Request limit must be a positive integer.')
    state = {'ready': False, 'failed': False}
    busy = threading.Lock()
    calls = deque()

    def warm():
        # PSEUDOCODE: load and exercise weights once; retain no warm-up result or user text.
        try:
            predictor.predict('emotion', '今天心情平静。')
            state['ready'] = True
        except Exception:
            import traceback
            traceback.print_exc()
            state['failed'] = True

    @asynccontextmanager
    async def lifespan(app):
        # PSEUDOCODE: let the page report loading while the model initializes; finish the worker on shutdown.
        task = asyncio.create_task(run_in_threadpool(warm))
        yield
        await task

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))
    app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins), allow_methods=['GET', 'POST'],
                       allow_headers=['Content-Type'], allow_credentials=False)

    @app.middleware('http')
    async def protect_origin(request, call_next):
        # PSEUDOCODE: reject cross-site writes and disable storage of input-bearing responses in browser caches.
        origin = request.headers.get('origin')
        same_origin = str(request.base_url).rstrip('/')
        if request.method == 'POST' and origin and origin not in (*allowed_origins, same_origin):
            return JSONResponse({'error': '不允许从这个网页提交。'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.get('/api/health')
    async def health():
        # PSEUDOCODE: expose readiness and public limits without filesystem paths, prompts or credentials.
        return {**state, 'busy': busy.locked(), 'max_characters': MAX_CHARACTERS,
                'model_id': predictor.model_identity, 'experimental': True, 'stores_text': False}

    @app.get('/api/schema')
    async def contract():
        # PSEUDOCODE: share the exact model category and metric definitions with the page.
        return schema()

    @app.post('/api/predict')
    async def predict(request: Request):
        # PSEUDOCODE: reject malformed/oversized requests before GPU work; retain the model's experimental status.
        if request.headers.get('content-type', '').split(';')[0].strip() != 'application/json':
            return JSONResponse({'error': '请使用 JSON 提交文本。'}, status_code=415)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse({'error': '提交内容太长，请缩短。'}, status_code=413)
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or set(data) != {'text', 'category'}:
                raise ValueError('请提交文本并选择一个类别。')
            category, text = validate_input(data['category'], data['text'])
            if len(text) > MAX_CHARACTERS:
                raise ValueError(f'最多输入{MAX_CHARACTERS}个字符，请分段分析。')
        except (ValueError, TypeError, UnicodeError) as error:
            return JSONResponse({'error': str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError)
                                 else '提交内容格式不正确。'}, status_code=422)
        if not state['ready']:
            message = '模型加载失败，请检查 CUDA、显存和模型文件。' if state['failed'] else '模型正在加载，请稍后重试。'
            return JSONResponse({'error': message}, status_code=503, headers={'Retry-After': '10'})
        now = time.monotonic()
        while calls and calls[0] <= now - 60:
            calls.popleft()
        if len(calls) >= requests_per_minute or not busy.acquire(blocking=False):
            return JSONResponse({'error': '当前服务繁忙，请稍后再试。'}, status_code=429, headers={'Retry-After': '5'})
        calls.append(now)
        try:
            result = await run_in_threadpool(predictor.predict, category, text)
            values = result.get('estimates', {})
            if set(values) != set(CATEGORIES[category][1]) or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 100 for v in values.values()):
                raise RuntimeError('Invalid model output.')
            return {'result': result, 'elapsed_seconds': round(time.monotonic() - now, 3)}
        except ValueError:
            return JSONResponse({'error': '这段文字超出模型处理长度或不符合要求，请缩短后重试。'}, status_code=422)
        except Exception:
            return JSONResponse({'error': '本次计算失败，请检查服务端显存和运行状态。'}, status_code=503)
        finally:
            busy.release()

    app.mount('/', StaticFiles(directory=STATIC, html=True), name='page')
    return app
