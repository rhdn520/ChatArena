import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from chatarena.agent import Player
from chatarena.arena import Arena
from chatarena.backends.base import IntelligenceBackend
from chatarena.backends.human import Human
from chatarena.environments.mafia import Mafia
from chatarena.ui.mafia_session import GameSession, SessionRegistry

NAMES = ["A", "B", "C", "D"]
ROLES = dict(zip(NAMES, ["mafia", "doctor", "police", "villager"]))


class MockBackend(IntelligenceBackend):
    type_name = "test-mafia-mock"
    stateful = False
    def __init__(self, entered=None, release=None):
        super().__init__()
        self.entered = entered
        self.release = release
    def query(self, agent_name, request_msg=None, **kwargs):
        instruction = request_msg.content
        if "REQUEST intent:" in instruction:
            return "3" if agent_name == "A" else "0"
        if "REQUEST speech:" in instruction:
            if self.entered:
                self.entered.set()
                self.release.wait(2)
            return "old generated text"
        if "NIGHT_MAFIA" in instruction:
            return "eliminate D"
        if "NIGHT_DOCTOR" in instruction:
            return "protect D"
        if "NIGHT_POLICE" in instruction:
            return "investigate A"
        return "vote A"


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out")


class TestMafiaSession(unittest.TestCase):
    def test_gradio_poll_and_pause_without_session(self):
        from gradio.state_holder import SessionState
        with patch("dotenv.load_dotenv"):
            import app

        async def check():
            state = SessionState(app.demo)
            for session_id in (None, "expired-session"):
                state[app.state._id] = {"session_id": session_id}
                for handler in (app.refresh_game, app.pause_game):
                    block_fn = next(fn for fn in app.demo.fns.values() if fn.fn is handler)
                    result = await app.demo.process_api(block_fn, [None], state=state)
                    self.assertEqual(len(result["data"]), len(app.live_outputs))
                    self.assertTrue(all(value == {"__type__": "update"}
                                        for value in result["data"]))

        asyncio.run(check())

    def arena(self, entered=None, release=None):
        players = [Player(n, "Play Mafia", Human() if n == "D" else MockBackend(entered, release)) for n in NAMES]
        return Arena(players, Mafia(NAMES, role_mapping=ROLES, max_discussion_messages=1))

    def test_web_session_interrupt_to_voting_and_private_view(self):
        entered, release = threading.Event(), threading.Event()
        arena = self.arena(entered, release)
        session = GameSession(arena)
        self.addCleanup(session.close)
        self.addCleanup(release.set)
        session.start()
        wait_for(entered.is_set)
        session.submit("Wait! A is lying.")
        snap = session.snapshot()
        self.assertEqual(snap["human"], "D")
        self.assertEqual(set(snap["views"]), {"D"})
        self.assertTrue(all(m.visible_to == "all" or "D" in m.visible_to for m in snap["messages"]))
        release.set()
        wait_for(lambda: arena.environment.phase == "DAY_VOTING")
        wait_for(lambda: session.status == "waiting_human")
        session.submit("vote A")
        wait_for(lambda: arena.environment.is_terminal())
        self.assertFalse(any(m.content == "old generated text" for m in arena.environment.get_observation()))

    def test_registry_clear_invalidates_old_session(self):
        registry = SessionRegistry()
        arena = self.arena()
        key = registry.add(arena)
        registry.remove(key)
        self.assertIsNone(registry.get(key))
        self.assertTrue(arena.discussion.closed)

    def test_example_configuration_survives_ui(self):
        # Prevent .env from enabling external API tests when the suite imports app.
        with patch("dotenv.load_dotenv"):
            import app
        values = {c: getattr(c, "value", None) for c in app.all_components}
        values[app.example_selector] = "Mafia"
        updates = app.update_components_from_example(values)
        for component, update in updates.items():
            values[component] = update.get("value") if isinstance(update, dict) else update
        values[app.mafia_options] = '{"seed": 17, "max_discussion_messages": 7}'
        config = app._create_arena_config_from_components(values)
        self.assertEqual(config.environment.seed, 17)
        self.assertEqual(config.environment.max_discussion_messages, 7)
        self.assertEqual(config.global_prompt, app.EXAMPLE_REGISTRY["Mafia"]["global_prompt"])
        self.assertIsNone(config.environment.discussion_moderator)

    def test_other_environment_still_steps_normally(self):
        from chatarena.environments.conversation import Conversation
        class ConversationBackend(MockBackend):
            def query(self, **kwargs):
                return "Hello"
        players = [Player(n, "chat", ConversationBackend()) for n in ("A", "B")]
        arena = Arena(players, Conversation(["A", "B"]))
        self.assertIsNone(arena.discussion)
        arena.step()
        self.assertEqual(arena.environment.get_next_player(), "B")
        arena.step()
        self.assertEqual(arena.environment.get_next_player(), "A")
        self.assertEqual(len(arena.current_timestep.observation), 2)

    def test_arena_reset_and_snapshot(self):
        arena = self.arena()
        self.addCleanup(arena.close)
        arena.step()
        self.assertEqual(arena.current_timestep, arena.environment.timestep())
        old = arena.environment.session_id
        arena.reset()
        self.assertNotEqual(old, arena.environment.session_id)
        self.assertEqual(arena.environment.phase, "NIGHT_MAFIA")
