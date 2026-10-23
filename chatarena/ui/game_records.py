"""Write completed UI games with message audiences for the history viewer."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from ..backends.human import Human


def generation_settings(backend):
    fields = ("temperature", "max_tokens", "enable_thinking", "thinking_token_budget",
              "control_thinking_token_budget", "use_stop", "system_messages_at_start",
              "retry_empty_response")
    settings = {key: getattr(backend, key) for key in fields if hasattr(backend, key)}
    for name in ("sampling_options", "sampling_extra"):
        settings.update({key: value for key, value in getattr(backend, name, {}).items()
                         if key in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")})
    return settings


def save_finished_game(arena, game, session_id, game_number=1):
    env = arena.environment
    if not env.is_terminal():
        return None
    root = Path(__file__).resolve().parents[2]
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = (root / "outputs" / "mixed_games" / f"job_{job}" / game
            / f"{session_id}-game-{game_number:04d}.json")
    record = {
        "game": game, "session_id": session_id, "game_number": game_number,
        "job_id": job, "finished_at": datetime.now(timezone.utc).isoformat(),
        "players": [{"name": p.name,
                     "type": "human" if isinstance(p.backend, Human) else "ai",
                     "model": None if isinstance(p.backend, Human) else getattr(p.backend, "model", None),
                     "settings": generation_settings(p.backend)}
                    for p in arena.players],
        "settings": {
            "environment": {key: getattr(env, key) for key in (
                "clue_rounds", "reveal_topic", "random_seed", "seed", "max_days",
                "reveal_role_on_death", "max_discussion_messages", "max_intent_rounds",
                "discussion_seconds", "silence_seconds", "intent_weights",
                "repeat_speaker_factor", "mafia_discussion_messages",
                "mafia_intent_rounds", "mafia_discussion_seconds") if hasattr(env, key)},
        },
        "rewards": env.get_rewards(),
        "message_scope": "all",
        "messages": [{"speaker": m.agent_name, "turn": m.turn,
                      "timestamp": m.timestamp, "type": m.msg_type,
                      "visible_to": m.visible_to, "content": m.content}
                     for m in env.message_pool.get_all_messages()],
    }
    if game == "mafia":
        moderator = env.discussion_moderator
        record["settings"]["moderator"] = {"enabled": bool(moderator)}
        if moderator:
            record["settings"]["moderator"].update({
                key: value for key, value in moderator.get("backend", {}).items()
                if key in ("model", "temperature", "max_tokens", "enable_thinking",
                           "thinking_token_budget", "control_thinking_token_budget",
                           "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")})
    if game == "liar":
        record["result"] = {name: getattr(env, name) for name in (
            "winner", "liar_name", "topic", "secret_word", "word_guess", "votes")}
        record["clue_rounds"] = env.clue_rounds
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        logging.info("GAME_SAVED game=%s session=%s game_number=%d path=%s roster=%s",
                     game, session_id, game_number, path, record["players"])
        return str(path)
    except OSError:
        logging.exception("GAME_SAVE_FAILED game=%s session=%s path=%s", game, session_id, path)
        return None
