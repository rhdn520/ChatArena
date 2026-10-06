import os
import hashlib
import logging
import threading
import re
from typing import List

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME, Message
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


def _retryable_error(exc):
    return isinstance(exc, openai.APIConnectionError) or (
        isinstance(exc, openai.APIStatusError)
        and (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
    )


# Share confirmed option rejections across player backends using the same
# client/model/settings. Protect only cache access, never the network call.
_CAPABILITY_LOCK = threading.Lock()
_UNSUPPORTED_OPTIONS = {}


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
            is_openai_available
        ), "openai package is not installed or the API key is not set"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self._token_parameter = "max_completion_tokens"
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

    @staticmethod
    def format_messages(agent_name, system_prompt, history_messages, request_msg=None):
        def participant_name(name):
            return "player_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:24]

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                messages.append({"role": "system", "content": msg.content})
            else:
                messages.append({
                    "role": "assistant" if msg.agent_name == agent_name else "user",
                    "name": participant_name(msg.agent_name),
                    "content": f"[{msg.agent_name}]: {msg.content}{END_OF_MESSAGE}",
                })
        messages.append({
            "role": "system",
            "content": request_msg.content if request_msg else f"Now you speak, {agent_name}.",
        })
        return messages

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60),
           retry=retry_if_exception(_retryable_error))
    def _get_response(self, messages):
        # Keep max_tokens in saved arena configs; translate at the API boundary.
        # Cache explicit capability rejections so subsequent turns avoid them.
        while True:
            options = {"temperature": self.temperature, "stop": STOP}
            capability_key = (client, self.model, self.temperature)
            with _CAPABILITY_LOCK:
                unsupported = set(_UNSUPPORTED_OPTIONS.get(capability_key, ()))
            for option in unsupported:
                options.pop(option, None)
            options[self._token_parameter] = self.max_tokens
            try:
                completion = client.chat.completions.create(
                    model=self.model, messages=messages, **options)
                break
            except openai.BadRequestError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                error = body.get("error", body)
                parameter = error.get("param")
                code = error.get("code")
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

        response = completion.choices[0].message.content
        if not response or not response.strip():
            reason = completion.choices[0].finish_reason
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
        if global_prompt:  # Prepend the global prompt if it exists
            system_prompt = f"You are a helpful assistant.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\nYour name is {agent_name}.\n\nYour role:{role_desc}"
        else:
            system_prompt = f"You are a helpful assistant. Your name is {agent_name}.\n\nYour role:{role_desc}\n\n{BASE_PROMPT}"

        messages = self.format_messages(agent_name, system_prompt, history_messages, request_msg)

        response = self._get_response(messages, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()  # noqa: F541
        response = re.sub(
            rf"^\s*{re.escape(agent_name)}\s*:", "", response
        ).strip()  # noqa: F451

        # Remove the tailing end of message token
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response
