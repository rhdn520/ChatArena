import os
import hashlib
import logging
import threading
import re
import time
import uuid
from urllib.parse import urlsplit
from typing import List

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME, Message, visibility_label
from .base import IntelligenceBackend, register_backend

try:
    import openai
except ImportError:
    is_openai_available = False
    # logging.warning("openai package is not installed")
else:
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        is_openai_available = True
    except openai.OpenAIError:
        # logging.warning("OpenAI API key is not set. Please set the environment variable OPENAI_API_KEY")
        is_openai_available = False

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gpt-3.5-turbo"
# DEFAULT_MODEL = "gpt-4-0613"

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."
FINAL_REASONING_TRANSITION = (
    "\nI have finished reasoning. I will now output only the requested final response, "
    "with no analysis, speaker prefix, or visibility labels.\n</think>\n\n"
)


def _retryable_error(exc):
    return isinstance(exc, openai.APIConnectionError) or (
        isinstance(exc, openai.APIStatusError)
        and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
    )


# Share confirmed option rejections across player backends using the same
# client/model/settings. Protect only cache access, never the network call.
_CAPABILITY_LOCK = threading.Lock()
_UNSUPPORTED_OPTIONS = {}


def _log_api_wait(done, started, call_id, player, model):
    while not done.wait(15):
        logging.info("API_WAIT id=%s player=%s model=%s elapsed=%.1fs",
                     call_id, player, model, time.monotonic() - started)


@register_backend
class OpenAIChat(IntelligenceBackend):
    """Interface to the ChatGPT style model with system, user, assistant roles separation."""

    stateful = False
    type_name = "openai-chat"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        merge_other_agents_as_one_user: bool = True,
        base_url: str = None,
        api_key_env: str = None,
        ssl_verify: bool = True,
        use_stop: bool = True,
        **kwargs,
    ):
        """
        Instantiate the OpenAIChat backend.

        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        assert (
            is_openai_available or (base_url and "openai" in globals())
        ), "openai package is not installed or the API key is not set"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            base_url=base_url,
            api_key_env=api_key_env,
            ssl_verify=ssl_verify,
            use_stop=use_stop,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        # Apply the known proxy limitation even when an older config omits use_stop.
        self.use_stop = use_stop and model.rsplit("/", 1)[-1] != "gpt-5.6-luna"
        self._custom_client = None
        if base_url:
            key = os.environ.get(api_key_env or "OPENAI_API_KEY")
            if not key:
                raise ValueError(f"Export {api_key_env or 'OPENAI_API_KEY'} before starting the game.")
            client_options = {}
            if not ssl_verify:
                client_options["http_client"] = openai.DefaultHttpxClient(verify=False)
            self._custom_client = openai.OpenAI(api_key=key, base_url=base_url, **client_options)
        self.sampling_options = {key: kwargs[key] for key in ("top_p", "presence_penalty")
                                 if kwargs.get(key) is not None}
        self.sampling_extra = {key: kwargs[key] for key in ("top_k", "min_p", "repetition_penalty")
                               if kwargs.get(key) is not None}
        self.enable_thinking = kwargs.get("enable_thinking")
        self.retry_empty_response = kwargs.get("retry_empty_response", False)
        self.thinking_token_budget = kwargs.get("thinking_token_budget")
        self.control_thinking_token_budget = kwargs.get("control_thinking_token_budget")
        for budget in (self.thinking_token_budget, self.control_thinking_token_budget):
            if budget is not None and (self.enable_thinking is not True
                                       or not isinstance(budget, int) or not 0 < budget < max_tokens):
                raise ValueError("Thinking budgets require thinking ON and must be between 0 and max_tokens")
        self.system_messages_at_start = kwargs.get(
            "system_messages_at_start",
            model.rsplit("/", 1)[-1] == "gpt-5.6-luna"
            or model.rsplit("/", 1)[-1].lower().startswith("qwen3.5-"),
        )
        self._token_parameter = "max_completion_tokens"
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

    @staticmethod
    def format_messages(agent_name, system_prompt, history_messages, request_msg=None,
                        system_messages_at_start=False, native_termination=False):
        def participant_name(name):
            return "player_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history_messages:
            content = f"{visibility_label(msg)} [{msg.agent_name}]: {msg.content}"
            if msg.agent_name == SYSTEM_NAME:
                messages.append({"role": "user" if system_messages_at_start else "system",
                                 "content": content})
            elif native_termination and msg.agent_name == agent_name:
                # Preserve the audience without teaching the model to reproduce
                # transcript headers as part of its next answer.
                messages.append({
                    "role": "user",
                    "content": "Previous reply metadata (not answer text): "
                               f"{visibility_label(msg)}. The next message is its text.",
                })
                messages.append({"role": "assistant", "name": participant_name(agent_name),
                                 "content": msg.content})
            else:
                messages.append({
                    "role": "assistant" if msg.agent_name == agent_name else "user",
                    "name": participant_name(msg.agent_name),
                    "content": content if native_termination else f"{content}{END_OF_MESSAGE}",
                })
        messages.append({
            "role": "user" if system_messages_at_start else "system",
            "content": request_msg.content if request_msg else f"Now you speak, {agent_name}.",
        })
        return messages

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60),
           retry=retry_if_exception(_retryable_error))
    def _get_response(self, messages, thinking_token_budget=None,
                      _final_continuation=False, _token_limit=None, _log_player=None,
                      _empty_retry=False):
        # Keep max_tokens in saved arena configs; translate at the API boundary.
        # Cache explicit capability rejections so subsequent turns avoid them.
        while True:
            options = {"temperature": self.temperature, **self.sampling_options}
            if self.model.rsplit("/", 1)[-1] == "gpt-5.6-luna":
                options.pop("temperature", None)
            # A reasoning model may quote a textual marker while still thinking.
            # Its native end-of-turn token must terminate generation instead.
            if self.use_stop and self.enable_thinking is not True:
                options["stop"] = STOP
            if self.enable_thinking is not None:
                options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
            budget = self.thinking_token_budget if thinking_token_budget is None else thinking_token_budget
            if budget is not None and not _final_continuation:
                options["extra_body"]["thinking_token_budget"] = budget
            if _final_continuation:
                # Reasoning is already in the assistant prefix. Qwen's
                # non-streaming parser otherwise labels a tag-free continuation
                # as reasoning again; do not open a second thinking section.
                options["extra_body"]["chat_template_kwargs"]["enable_thinking"] = False
                options["extra_body"].update(continue_final_message=True,
                                             add_generation_prompt=False)
            if self.sampling_extra:
                options.setdefault("extra_body", {}).update(self.sampling_extra)
            api_client = self._custom_client or client
            capability_key = (api_client, self.model, self.temperature)
            with _CAPABILITY_LOCK:
                unsupported = set(_UNSUPPORTED_OPTIONS.get(capability_key, ()))
            for option in unsupported:
                options.pop(option, None)
            options[self._token_parameter] = self.max_tokens if _token_limit is None else _token_limit
            call_id = uuid.uuid4().hex[:8]
            started = time.monotonic()
            endpoint = urlsplit(str(getattr(api_client, "base_url", "")))
            logging.info("API_START id=%s player=%s model=%s host=%s port=%s messages=%d system_at=%s params=%s",
                         call_id, _log_player, self.model, endpoint.hostname, endpoint.port,
                         len(messages), [i for i, m in enumerate(messages) if m.get("role") == "system"],
                         sorted(options))
            done = threading.Event()
            threading.Thread(target=_log_api_wait,
                             args=(done, started, call_id, _log_player, self.model),
                             daemon=True).start()
            try:
                completion = api_client.chat.completions.create(
                    model=self.model, messages=messages, **options)
                logging.info("API_DONE id=%s player=%s model=%s elapsed=%.1fs finish=%s completion_tokens=%s",
                             call_id, _log_player, self.model, time.monotonic() - started,
                             completion.choices[0].finish_reason if completion.choices else None,
                             getattr(completion.usage, "completion_tokens", None))
                break
            except openai.BadRequestError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                error = body.get("error", body)
                parameter = error.get("param")
                code = error.get("code")
                logging.warning("API_REJECT id=%s player=%s model=%s elapsed=%.1fs param=%s code=%s",
                                call_id, _log_player, self.model, time.monotonic() - started,
                                parameter, code)
                # LiteLLM reports unsupported options as code '400', param=None.
                # Normalize only the explicitly listed options handled below.
                if code not in ("unsupported_parameter", "unsupported_value"):
                    detail = str(error.get("message", ""))
                    rejected = re.search(r"does not support parameters:\s*\[([^\]]*)\]", detail)
                    if "litellm.UnsupportedParamsError" in detail and rejected:
                        names = re.findall(r"['\"]([a-z_]+)['\"]", rejected.group(1))
                        parameter = next((name for name in names
                                          if name in ("stop", "temperature", "max_completion_tokens")
                                          and name in options), None)
                        if parameter is not None:
                            code = "unsupported_parameter"
                    elif (parameter in ("temperature", "stop") and parameter in options
                          and "Unsupported value:" in detail):
                        code = "unsupported_value"
                if code not in ("unsupported_parameter", "unsupported_value"):
                    raise
                if parameter == "max_completion_tokens" and self._token_parameter == parameter:
                    self._token_parameter = "max_tokens"
                elif parameter in ("temperature", "stop") and parameter in options:
                    with _CAPABILITY_LOCK:
                        unsupported = _UNSUPPORTED_OPTIONS.setdefault(capability_key, set())
                        first_rejection = parameter not in unsupported
                        unsupported.add(parameter)
                    if first_rejection:
                        logging.warning("Model %s rejected %s; using its default behavior "
                                        "for all players with these settings",
                                        self.model, parameter)
                else:
                    raise

            except openai.APIConnectionError as exc:
                logging.warning("API_ERROR id=%s player=%s model=%s elapsed=%.1fs connection=%s",
                                call_id, _log_player, self.model, time.monotonic() - started,
                                exc.__cause__ or exc)
                raise
            except Exception as exc:
                logging.error("API_ERROR id=%s player=%s model=%s elapsed=%.1fs type=%s status=%s",
                              call_id, _log_player, self.model, time.monotonic() - started,
                              type(exc).__name__, getattr(exc, "status_code", None))
                raise
            finally:
                done.set()

        message = completion.choices[0].message
        response = message.content or ""
        reasoning = (getattr(message, "reasoning", None)
                     or getattr(message, "reasoning_content", None))
        logging.info("API_TEXT id=%s player=%s model=%s content_chars=%d reasoning_chars=%d stop_reason=%s",
                     call_id, _log_player, self.model, len(response), len(reasoning or ""),
                     getattr(completion.choices[0], "stop_reason", None))
        if self.enable_thinking is True:
            # The API already separates reasoning. If another boundary appears
            # in content, keep only its final segment, never the reasoning field.
            response = response.rsplit("</think>", 1)[-1]
            response = response.split("<think>", 1)[0]
        if not response or not response.strip():
            reason = completion.choices[0].finish_reason
            logging.warning("API_EMPTY id=%s player=%s model=%s finish=%s completion_tokens=%s retry=%s",
                            call_id, _log_player, self.model, reason,
                            getattr(completion.usage, "completion_tokens", None), _empty_retry)
            used = completion.usage.completion_tokens if completion.usage else self.max_tokens
            remaining = self.max_tokens - used
            if (self.enable_thinking is True and not _final_continuation
                    and reasoning and remaining > 0):
                logging.warning("Model %s returned no final text (%s); continuing once "
                                "after its private reasoning, with %s tokens remaining",
                                self.model, reason, remaining)
                continuation_messages = [*messages, {
                    "role": "assistant",
                    "content": "<think>\n" + reasoning.strip() + FINAL_REASONING_TRANSITION,
                }]
                return self._get_response(continuation_messages, thinking_token_budget,
                                          _final_continuation=True, _token_limit=remaining,
                                          _log_player=_log_player, _empty_retry=_empty_retry)
            if self.retry_empty_response and not _empty_retry and reason == "stop":
                logging.warning("API_EMPTY_RETRY player=%s model=%s attempt=1/1",
                                _log_player, self.model)
                return self._get_response(messages, thinking_token_budget,
                                          _final_continuation=_final_continuation,
                                          _token_limit=_token_limit, _log_player=_log_player,
                                          _empty_retry=True)
            if reason == "length":
                suggested_limit = max(4096, self.max_tokens * 2)
                raise ValueError(
                    f"Model {self.model} returned no text (finish_reason=length, "
                    f"token_limit={self.max_tokens}). Please increase the token limit. "
                    f"UI의 'AI 응답 설정 → 최대 생성 토큰'을 {suggested_limit} 이상으로 "
                    "늘린 뒤 초기화하여 새 게임을 시작해주세요. 내부 추론도 한도에 포함됩니다."
                )
            raise ValueError(f"Model {self.model} returned no text (finish_reason={reason}).")
        return response.strip()

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Format the input and call the ChatGPT/GPT-4 API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            env_desc: the description of the environment
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """

        # Merge the role description and the global prompt as the system prompt for the agent
        native_termination = self.enable_thinking is True
        base_prompt = "" if native_termination else BASE_PROMPT
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"You are a helpful assistant.\n{global_prompt.strip()}\n{base_prompt}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            system_prompt = f"You are a helpful assistant. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{base_prompt}"

        messages = self.format_messages(agent_name, system_prompt, history_messages, request_msg,
                                        system_messages_at_start=self.system_messages_at_start,
                                        native_termination=native_termination)

        budget = self.thinking_token_budget
        if request_msg and request_msg.content.startswith(("REQUEST intent:", "REQUEST moderation:")):
            budget = self.control_thinking_token_budget or budget
        response = self._get_response(messages, *args, thinking_token_budget=budget,
                                      _log_player=agent_name, **kwargs)

        # Strip transcript metadata only; a bracketed vote target is content.
        speaker = rf"(?:\[{re.escape(agent_name)}\]|{re.escape(agent_name)})\s*:\s*"
        audience = r"\[(?:Public|Private)(?::[^\]\r\n]*)?\]\s*:?\s*"
        response = re.sub(
            rf"^\s*(?:{audience}|{speaker})+", "", response, flags=re.IGNORECASE
        ).strip()

        # Remove the tailing end of message token
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response
