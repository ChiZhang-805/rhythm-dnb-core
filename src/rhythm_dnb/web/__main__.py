"""Start the same browser application on a local NVIDIA computer or an explicit GPU host."""

import argparse
import os


def main():
    # PSEUDOCODE: require local weights and CUDA -> bind locally by default -> start exactly one model worker.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='models/rhythm-text-expanded')
    parser.add_argument('--base', default='models/qwen3-8b')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=7860, type=int)
    parser.add_argument('--allow-origin', action='append', default=[])
    parser.add_argument('--allow-host', action='append', default=[])
    parser.add_argument('--requests-per-minute', default=30, type=int)
    parser.add_argument('--precision', choices=('fp32', 'auto'), default='fp32')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    if not torch.cuda.is_available():
        parser.error('当前 NF4 模型需要 NVIDIA GPU 和 CUDA 版 PyTorch；仅有网页或 CPU 不能运行。')
    torch.set_num_threads(4)
    from ..text.predict import TextPredictor
    from .server import create_app
    predictor = TextPredictor(args.checkpoint, base_path=args.base, device='cuda', allow_experimental=True, precision=args.precision)
    app = create_app(predictor, allowed_origins=args.allow_origin,
                     allowed_hosts=['localhost', '127.0.0.1', '[::1]', *args.allow_host],
                     requests_per_minute=args.requests_per_minute)
    import uvicorn
    print(f'打开 http://{args.host}:{args.port} ，等待模型加载。按 Ctrl+C 退出。', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False,
                proxy_headers=False, limit_concurrency=40, timeout_keep_alive=5)


if __name__ == '__main__':
    main()
