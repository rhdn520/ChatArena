import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import httpx
import openai
from tenacity import wait_none
from chatarena.backends import openai as backend
from chatarena.message import Message, SYSTEM_NAME


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

    def test_thinking_uses_native_termination_and_per_request_budgets(self):
        with patch.object(backend, 'is_openai_available', True):
            model = backend.OpenAIChat(model='Qwen/Qwen3.5-9B', max_tokens=4096,
                                       enable_thinking=True, thinking_token_budget=2048,
                                       control_thinking_token_budget=256)
        response = completion('0')
        response.choices[0].message.reasoning = 'The prior <EOS> convention is irrelevant.'
        self.create.return_value = response
        history = [Message('Player 1', 'Previous speech', 0), Message('Player 2', 'Reply', 1)]
        for kind, budget in [('intent', 256), ('moderation', 256), ('speech', 2048),
                             ('game_action', 2048)]:
            self.assertEqual(model.query('Player 1', 'Your role', history,
                request_msg=Message(SYSTEM_NAME, f'REQUEST {kind}: Output 0', 2)), '0')
            args = self.create.call_args.kwargs
            self.assertNotIn('stop', args)
            self.assertTrue(all('<EOS>' not in m['content'] for m in args['messages']))
            self.assertEqual(args['extra_body'], {
                'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': budget})
            self.assertEqual(args['max_completion_tokens'], 4096)

    def test_thinking_does_not_accept_truncated_final_output(self):
        self.model.enable_thinking = True
        self.create.return_value = completion('Partial speech', 'length')
        with self.assertRaisesRegex(ValueError, 'partial final output'):
            self.model._get_response([])
        self.assertEqual(self.create.call_count, 1)

    def test_thinking_budget_rejection_cannot_be_silently_ignored(self):
        self.model.enable_thinking = True
        self.model.thinking_token_budget = 256
        self.create.side_effect = bad_request('thinking_token_budget')
        with self.assertRaises(openai.BadRequestError):
            self.model._get_response([])
        self.assertEqual(self.create.call_count, 1)

    def test_invalid_thinking_budget_is_rejected(self):
        for thinking, budget in [(False, 256), (True, 0), (True, 4096), (True, '256')]:
            with self.subTest(thinking=thinking, budget=budget), \
                    patch.object(backend, 'is_openai_available', True):
                with self.assertRaises(ValueError):
                    backend.OpenAIChat(max_tokens=4096, enable_thinking=thinking,
                                       thinking_token_budget=budget)

    def test_local_sampling_parameters_are_sent_without_silent_fallback(self):
        with patch.object(backend, 'is_openai_available', True):
            model = backend.OpenAIChat(model='Qwen/Qwen3.5-9B', max_tokens=4096,
                enable_thinking=True, thinking_token_budget=2048, temperature=1.0,
                top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0)
        self.create.return_value = completion('대사입니다.')
        model._get_response([])
        args = self.create.call_args.kwargs
        self.assertEqual((args['temperature'], args['top_p'], args['presence_penalty']), (1.0, 0.95, 1.5))
        self.assertEqual(args['extra_body'], {'chat_template_kwargs': {'enable_thinking': True},
            'thinking_token_budget': 2048, 'top_k': 20, 'min_p': 0.0, 'repetition_penalty': 1.0})
        self.create.side_effect = bad_request('presence_penalty')
        with self.assertRaises(openai.BadRequestError):
            model._get_response([])

    def test_reasoning_tag_in_final_content_is_not_a_game_action(self):
        self.model.enable_thinking = True
        self.create.return_value = completion('\n</think>\n\n3')
        with self.assertRaisesRegex(ValueError, 'boundary leaked'):
            self.model._get_response([])
