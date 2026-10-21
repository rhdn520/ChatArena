"""Seven shuffled seats shared by the human-facing Liar and Mafia UIs."""
import os
import random


PLAYER_COUNT = 7
DEFAULT_GPT_MODEL = "openai/gpt-5.6-luna"


def make_gpt_backend(temperature=0.7, max_tokens=4096, model=None):
    model = model or os.environ.get("CHAT_ARENA_GPT_MODEL", DEFAULT_GPT_MODEL)
    return {
        "backend_type": "openai-chat",
        "model": model,
        "base_url": os.environ.get("CHAT_ARENA_GPT_BASE_URL", "https://147.47.200.198:7861"),
        "api_key_env": os.environ.get("CHAT_ARENA_GPT_API_KEY_ENV", "SKIML_API_KEY"),
        "use_stop": False,
        "system_messages_at_start": True,
        "ssl_verify": os.environ.get("CHAT_ARENA_GPT_SSL_VERIFY", "false").lower() not in ("0", "false", "no"),
        "temperature": 1.0 if model.rsplit("/", 1)[-1] == "gpt-5.6-luna" else float(temperature),
        "max_tokens": int(max_tokens),
    }


def make_mixed_players(global_prompt, temperature=0.7, max_tokens=4096,
                       gpt_model=None, human_name=None):
    player_names = [f"Player {i}" for i in range(1, PLAYER_COUNT + 1)]
    rng = random.SystemRandom()
    if human_name is None:
        human_name = rng.choice(player_names)
    if human_name not in player_names:
        raise ValueError("Human seat must be Player 1 through Player 7.")
    backends = []
    for size, port in (("2B", 8001), ("9B", 8002)):
        default_model = "Qwen/Qwen3.5-2B" if size == "2B" else "Qwen/Qwen3.5-9B"
        backend = {
            "backend_type": "openai-chat",
            "model": os.environ.get(f"CHAT_ARENA_QWEN_{size}_MODEL", default_model),
            "base_url": os.environ.get(f"CHAT_ARENA_QWEN_{size}_BASE_URL", f"http://127.0.0.1:{port}/v1"),
            "api_key_env": "CHAT_ARENA_QWEN_API_KEY",
            "system_messages_at_start": True,
            # Let Qwen end its own turn; transcript markers can otherwise
            # truncate generation before the final answer is emitted.
            "use_stop": False,
            # Qwen3.5's template defaults to thinking off; the vLLM parser
            # defaults to on unless the same choice is explicitly supplied.
            "enable_thinking": False,
            "retry_empty_response": True,
            "temperature": float(temperature), "max_tokens": int(max_tokens),
        }
        backends.extend([backend.copy(), backend.copy()])
    gpt = make_gpt_backend(temperature, max_tokens, gpt_model)
    backends.extend([gpt.copy(), gpt.copy()])
    rng.shuffle(backends)
    ai_backends = iter(backends)
    return [{
        "name": name,
        "role_desc": f"당신은 {name}입니다. 자신의 팀이 승리하도록 행동하세요.",
        "global_prompt": global_prompt,
        "backend": {"backend_type": "human"} if name == human_name else next(ai_backends),
    } for name in player_names]
