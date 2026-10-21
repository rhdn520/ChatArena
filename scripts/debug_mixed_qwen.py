"""Explicit user-requested reproduction against already running local servers.

Only the isolated reproduction uses controlled seats. Production seats/prompts
are unchanged. This does not exercise the browser or call GPT.
"""
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["SKIML_API_KEY"] = "not-used-debug-blocks-remote"
os.environ["CHAT_ARENA_QWEN_API_KEY"] = "local-model"

from chatarena.backends.openai import OpenAIChat
from chatarena.ui.liar_game_gradio import create_human_game

logging.basicConfig(level=logging.INFO, force=True,
                    format="%(asctime)s %(levelname)s %(message)s")
original_query = OpenAIChat.query


def local_query(self, *args, **kwargs):
    endpoint = urlsplit(str(self._custom_client.base_url))
    if endpoint.hostname != "127.0.0.1" or endpoint.port not in (8001, 8002):
        raise RuntimeError("DEBUG_BLOCKED_REMOTE_API")
    self._custom_client = self._custom_client.with_options(timeout=90, max_retries=0)
    return original_query(self, *args, **kwargs)


OpenAIChat.query = local_query
failures = 0
for model in ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-9B"):
    for trial in range(1, 3):
        try:
            game = create_human_game("examples/liar_game.json", 1, 6, True,
                                     clue_rounds=1, human_names=["Player 7"])
            backend = next(p.backend for p in game.arena.players
                           if getattr(p.backend, "model", None) == model)
            # Control only this isolated reproduction's first two speakers.
            for name in ("Player 1", "Player 2"):
                game.arena.name_to_player[name].backend = backend
            logging.info("DEBUG_BEGIN model=%s trial=%d use_stop=%s thinking=%s",
                         model, trial, backend.use_stop, backend.enable_thinking)
            for turn in range(1, 3):
                if not game.step_ai_once() or game.timestep.terminal:
                    raise RuntimeError("AI did not advance a normal speaking turn")
                logging.info("DEBUG_TURN_OK model=%s trial=%d turn=%d next=%s",
                             model, trial, turn, game.environment.get_next_player())
        except Exception:
            failures += 1
            logging.exception("DEBUG_FAILED model=%s trial=%d", model, trial)
logging.info("DEBUG_RESULT failed_scenarios=%d total_scenarios=4 GPT_calls=0", failures)
raise SystemExit(1 if failures else 0)
