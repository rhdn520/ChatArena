import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import httpx
import openai
from tenacity import wait_none
from chatarena.backends import openai as backend


def bad_request(parameter, code='unsupported_parameter'):
    response = httpx.Response(400, request=httpx.Request('POST', 'https://example.test'))
    return openai.BadRequestError('Unsupported option', response=response,
                                 body={'param': parameter, 'code': code})


def completion(content='hello', finish_reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class TestOpenAIParameters(unittest.TestCase):
    def setUp(self):
        backend._UNSUPPORTED_OPTIONS.clear()
        self.client = Mock()
        self.patch = patch.object(backend, 'client', self.client, create=True)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        with patch.object(backend, 'is_openai_available', True):
            self.model = backend.OpenAIChat(model='gpt-6-luna', max_tokens=512)
        self.create = self.client.chat.completions.create

    def test_modern_token_parameter(self):
        self.create.return_value = completion()
        self.assertEqual(self.model._get_response([]), 'hello')
        args = self.create.call_args.kwargs
        self.assertEqual(args['model'], 'gpt-6-luna')
        self.assertEqual(args['max_completion_tokens'], 512)
        self.assertNotIn('max_tokens', args)

    def test_rejected_options_are_removed_and_cached(self):
        self.create.side_effect = [bad_request('temperature', 'unsupported_value'),
                                   bad_request('stop'), completion(), completion()]
        self.model._get_response([])
        self.model._get_response([])
        self.assertEqual(self.create.call_count, 4)
        for args in self.create.call_args_list[2:]:
            self.assertNotIn('temperature', args.kwargs)
            self.assertNotIn('stop', args.kwargs)
            self.assertEqual(args.kwargs['max_completion_tokens'], 512)

    def test_legacy_fallback_retains_limit(self):
        self.create.side_effect = [bad_request('max_completion_tokens'), completion(), completion()]
        self.model._get_response([])
        self.model._get_response([])
        self.assertEqual(self.create.call_args.kwargs['max_tokens'], 512)
        self.assertNotIn('max_completion_tokens', self.create.call_args.kwargs)

    def test_capabilities_shared_between_players_and_new_games(self):
        self.create.side_effect = [bad_request('temperature', 'unsupported_value'),
                                   bad_request('stop'), completion(), completion()]
        self.model._get_response([])
        with patch.object(backend, 'is_openai_available', True):
            other = backend.OpenAIChat(model='gpt-6-luna', max_tokens=4096)
        other._get_response([])
        args = self.create.call_args.kwargs
        self.assertNotIn('temperature', args)
        self.assertNotIn('stop', args)
        self.assertEqual(args['max_completion_tokens'], 4096)

    def test_capability_cache_does_not_affect_other_models(self):
        self.create.side_effect = [bad_request('temperature'), completion(), completion()]
        self.model._get_response([])
        with patch.object(backend, 'is_openai_available', True):
            other = backend.OpenAIChat(model='gpt-3.5-turbo')
        other._get_response([])
        self.assertIn('temperature', self.create.call_args.kwargs)

    def test_bad_requests_are_not_retried(self):
        self.create.side_effect = bad_request('messages', 'invalid_value')
        with self.assertRaises(openai.BadRequestError):
            self.model._get_response([])
        self.assertEqual(self.create.call_count, 1)

    def test_fallback_cannot_loop(self):
        self.create.side_effect = [bad_request('max_completion_tokens'), bad_request('max_tokens')]
        with self.assertRaises(openai.BadRequestError):
            self.model._get_response([])
        self.assertEqual(self.create.call_count, 2)

    def test_transient_failure_retries(self):
        self.create.side_effect = [openai.APIConnectionError(
            request=httpx.Request('POST', 'https://example.test')), completion()]
        call = self.model._get_response.retry_with(wait=wait_none())
        self.assertEqual(call(self.model, []), 'hello')
        self.assertEqual(self.create.call_count, 2)

    def test_empty_output_has_actionable_error_without_retry(self):
        self.create.return_value = completion('', 'length')
        with self.assertRaisesRegex(ValueError, 'increase the token limit'):
            self.model._get_response([])
        self.assertEqual(self.create.call_count, 1)
        self.assertEqual(self.create.call_args.kwargs['max_completion_tokens'], 512)
