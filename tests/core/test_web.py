"""Exercise browser/API boundaries without downloading weights or fabricating model quality evidence."""

import threading
import time
import unittest

try:
    from fastapi.testclient import TestClient
    from rhythm_dnb.web.server import create_app
except ImportError:
    TestClient = None

from rhythm_dnb.text.schema import CATEGORIES


class FakePredictor:
    model_identity = 'unit-test-only'

    def __init__(self):
        self.calls = []
        self.block = None
        self.fail = None

    def predict(self, category, text):
        self.calls.append((category, text))
        if self.block:
            self.block.wait(5)
        if self.fail:
            raise self.fail
        return {'category': category, 'estimates': {k: 25.25 for k in CATEGORIES[category][1]},
                'scores': {k: None for k in CATEGORIES[category][1]}, 'eligible_for_primary_dnb': False}


@unittest.skipIf(TestClient is None, 'Optional web dependencies are not installed.')
class WebTests(unittest.TestCase):
    def client(self, predictor, **kwargs):
        return TestClient(create_app(predictor, **kwargs), base_url='http://localhost')

    def ready(self, client):
        for _ in range(100):
            if client.get('/api/health').json()['ready']:
                return
            time.sleep(.01)
        self.fail('Model did not warm up.')

    def test_real_contract_and_raw_experimental_status_are_preserved(self):
        predictor = FakePredictor()
        with self.client(predictor) as client:
            self.ready(client)
            self.assertIn('文本观察室', client.get('/').text)
            self.assertEqual(len(client.get('/api/schema').json()['categories']), 5)
            for category in CATEGORIES:
                result = client.post('/api/predict', json={'category': category, 'text': '今天心情不错。'})
                self.assertEqual(result.status_code, 200)
                self.assertFalse(result.json()['result']['eligible_for_primary_dnb'])
                self.assertTrue(all(v is None for v in result.json()['result']['scores'].values()))
                self.assertEqual(result.headers['cache-control'], 'no-store')
            self.assertEqual(client.get('/models/manifest.json').status_code, 404)

    def test_invalid_requests_never_reach_the_predictor(self):
        predictor = FakePredictor()
        with self.client(predictor) as client:
            self.ready(client)
            before = len(predictor.calls)
            for data in ({'text': '中文'}, {'text': '中文', 'category': 'unknown'}, {'text': 'hello', 'category': 'emotion'},
                         {'text': '中' * 501, 'category': 'emotion'}, {'text': '', 'category': 'emotion'},
                         {'text': '中文', 'category': 'emotion', 'checkpoint': '/tmp/other'}, []):
                self.assertEqual(client.post('/api/predict', json=data).status_code, 422)
            self.assertEqual(client.post('/api/predict', content=b'X' * 8193, headers={'Content-Type': 'application/json'}).status_code, 413)
            self.assertEqual(client.post('/api/predict', content='text').status_code, 415)
            self.assertEqual(len(predictor.calls), before)

    def test_origin_and_host_checks_prevent_unwanted_browser_calls(self):
        predictor = FakePredictor()
        with self.client(predictor, allowed_origins=['https://chizhang-805.github.io']) as client:
            self.ready(client)
            data = {'text': '中文记录', 'category': 'emotion'}
            self.assertEqual(client.post('/api/predict', json=data, headers={'Origin': 'https://untrusted.test'}).status_code, 403)
            self.assertEqual(client.post('/api/predict', json=data, headers={'Host': 'rebind.test'}).status_code, 400)
            accepted = client.post('/api/predict', json=data, headers={'Origin': 'https://chizhang-805.github.io'})
            self.assertEqual(accepted.status_code, 200)
            self.assertEqual(accepted.headers['access-control-allow-origin'], 'https://chizhang-805.github.io')

    def test_requests_are_bounded_and_errors_do_not_leak_paths(self):
        predictor = FakePredictor()
        with self.client(predictor, requests_per_minute=1) as client:
            self.ready(client)
            predictor.fail = RuntimeError('private /path/to/model')
            result = client.post('/api/predict', json={'text': '中文记录', 'category': 'emotion'})
            self.assertEqual(result.status_code, 503)
            self.assertNotIn('private', result.text)
            predictor.fail = None
            self.assertEqual(client.post('/api/predict', json={'text': '中文记录', 'category': 'emotion'}).status_code, 429)

    def test_concurrent_requests_do_not_start_multiple_gpu_jobs(self):
        predictor = FakePredictor()
        with self.client(predictor) as client:
            self.ready(client)
            predictor.block = threading.Event()
            data = {'text': '中文记录', 'category': 'emotion'}
            worker = threading.Thread(target=lambda: client.post('/api/predict', json=data))
            worker.start()
            try:
                for _ in range(100):
                    if client.get('/api/health').json()['busy']:
                        break
                    time.sleep(.01)
                self.assertEqual(client.post('/api/predict', json=data).status_code, 429)
            finally:
                predictor.block.set()
                worker.join(10)
            self.assertFalse(client.get('/api/health').json()['busy'])

    def test_loading_model_has_no_fake_results(self):
        predictor = FakePredictor()
        predictor.block = threading.Event()
        with self.client(predictor) as client:
            try:
                result = client.post('/api/predict', json={'text': '中文记录', 'category': 'emotion'})
                self.assertEqual(result.status_code, 503)
                self.assertNotIn('result', result.json())
            finally:
                predictor.block.set()
