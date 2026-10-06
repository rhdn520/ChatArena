import asyncio
import unittest

from gradio.state_holder import SessionState

from chatarena.agent import Player
from chatarena.arena import Arena
from chatarena.backends.base import IntelligenceBackend
from chatarena.backends.human import Human
from chatarena.environments.mafia import Mafia
from chatarena.message import Message
from chatarena.ui.mafia_app import MafiaUI, make_config, MODERATOR_GUIDE


class QuietBackend(IntelligenceBackend):
    type_name = "test-mafia-ui"
    stateful = False

    def __init__(self):
        super().__init__()

    def query(self, request_msg=None, **kwargs):
        text = request_msg.content
        if "REQUEST intent:" in text:
            return "0"
        if "NIGHT_MAFIA" in text:
            return "eliminate Player 4"
        if "NIGHT_DOCTOR" in text:
            return "protect Player 4"
        if "NIGHT_POLICE" in text:
            return "investigate Player 1"
        return "vote Player 1"


def fake_arena(config):
    names = [p.name for p in config.players]
    roles = dict(zip(names, ["mafia", "doctor", "police", "villager"]))
    players = [Player(p.name, p.role_desc, Human() if p.backend.backend_type == "human"
                      else QuietBackend()) for p in config.players]
    options = dict(config.environment)
    options.pop("env_type")
    return Arena(players, Mafia(names, role_mapping=roles, **options))


class TestMafiaUI(unittest.TestCase):
    def setUp(self):
        self.ui = MafiaUI(arena_factory=fake_arena)
        self.state = SessionState(self.ui.demo)
        self.client_key = ""
        self.values = [component.value for component in self.ui.settings]

    def tearDown(self):
        for key in list(self.ui.registry.sessions):
            self.ui.registry.remove(key)
        self.ui.demo.close()

    async def call(self, handler, inputs):
        polling = handler == self.ui.refresh
        if polling:
            handler = self.ui.poll
        block_fn = next(fn for fn in self.ui.demo.fns.values() if fn.fn == handler)
        inputs = list(inputs)
        for i, component in enumerate(block_fn.inputs):
            if component is self.ui.key and inputs[i] is None:
                inputs[i] = self.client_key
        result = await self.ui.demo.process_api(block_fn, inputs, state=self.state)
        if self.ui.key in block_fn.outputs:
            value = result["data"][block_fn.outputs.index(self.ui.key)]
            if not isinstance(value, dict):
                self.client_key = value
        self.assertEqual(len(result["data"]), len(block_fn.outputs))
        if polling:
            result["data"] = result["data"][0].model_dump()
        return result

    def test_idle_pause_send_and_clear_through_gradio(self):
        async def scenario():
            for _ in range(2):
                result = await self.call(self.ui.refresh, [None])
                self.assertTrue(all(x == {"__type__": "update"} for x in result["data"]))
                await self.call(self.ui.pause, [None])
                await self.call(self.ui.send, [None, "hello"])
                await self.call(self.ui.clear, [None])
        asyncio.run(scenario())

    def test_play_start_input_pause_and_clear(self):
        async def scenario():
            await self.call(self.ui.start, [None, *self.values])
            key = self.client_key
            session = self.ui.registry.get(key)
            self.assertEqual(session.human, "Player 1")
            await self.call(self.ui.send, [None, "eliminate Player 4"])
            for _ in range(200):
                if session.arena.environment.phase == "DAY_DISCUSSION":
                    break
                await asyncio.sleep(.01)
            self.assertEqual(session.arena.environment.phase, "DAY_DISCUSSION")
            await self.call(self.ui.send, [None, "누가 의심스러운가요?"])
            await self.call(self.ui.pause, [None])
            result = await self.call(self.ui.refresh, [None])
            self.assertIn("마피아", result["data"][1])
            chat = str(result["data"][3])
            self.assertNotIn("당신의 비밀 역할은 **경찰**", chat)
            self.assertIn("누가 의심스러운가요?", chat)
            await self.call(self.ui.clear, [None])
            self.assertIsNone(self.ui.registry.get(key))
            await self.call(self.ui.refresh, [None])
        asyncio.run(scenario())

    def test_browser_inputs_match_advertised_endpoint_parameters(self):
        config = self.ui.demo.get_config_file()
        info = self.ui.demo.get_api_info()
        for dep in config["dependencies"]:
            if dep["backend_fn"]:
                endpoint = info["named_endpoints"]["/" + dep["api_name"]]
                self.assertEqual(len(dep["inputs"]), len(endpoint["parameters"]),
                                 dep["api_name"])
                self.assertEqual(info["unnamed_endpoints"][str(dep["id"])], endpoint)

    def test_chat_has_no_backend_event_outputs(self):
        # Suppressing a spinner does not suppress Gradio Chatbot.pending_message.
        # No server request (including idle polling) may target the Chatbot.
        config = self.ui.demo.get_config_file()
        writers = [dep for dep in config["dependencies"]
                   if self.ui.chat._id in dep["outputs"]]
        self.assertEqual(len(writers), 1)
        self.assertFalse(writers[0]["backend_fn"])
        self.assertEqual(writers[0]["inputs"], [self.ui.chat_payload._id])
        self.assertEqual(writers[0]["js"], "(messages) => [messages || []]")
        poll = next(fn for fn in self.ui.demo.fns.values() if fn.fn == self.ui.poll)
        self.assertEqual(poll.outputs, [self.ui.poll_payload])
        self.assertFalse(self.ui.poll_payload.visible)
        self.assertTrue(all(comp not in poll.outputs for comp in self.ui.live_outputs))
        self.assertFalse(self.ui.chat.show_label)

    def test_invalid_settings_returns_full_outputs(self):
        values = list(self.values)
        values[2] = ""
        result = asyncio.run(self.call(self.ui.start, [None, *values]))
        self.assertIn("모델 이름", result["data"][-1])
        self.assertFalse(self.ui.registry.sessions)

    def test_poll_skips_unchanged_chat_and_updates_new_messages(self):
        async def scenario():
            await self.call(self.ui.start, [None, *self.values])
            key = self.client_key
            session = self.ui.registry.get(key)
            await self.call(self.ui.pause, [None])
            first = await self.call(self.ui.refresh, [None])
            self.assertIsInstance(first["data"][3], list)
            second = await self.call(self.ui.refresh, [None])
            self.assertTrue(all(x == {"__type__": "update"} for x in second["data"]))
            with session.lock:
                session.arena.environment._moderator_speak("새 공개 메시지")
            updated = await self.call(self.ui.refresh, [None])
            self.assertIn("새 공개 메시지", str(updated["data"][3]))
            self.assertEqual(updated["data"][1], {"__type__": "update"})
            with session.lock:
                session.arena.environment._moderator_speak("다른 사람의 비밀", visible_to=["Player 2"])
            private = await self.call(self.ui.refresh, [None])
            self.assertEqual(private["data"][3], {"__type__": "update"})
            poll_fn = next(fn for fn in self.ui.demo.fns.values() if fn.fn == self.ui.poll)
            self.assertEqual(poll_fn.show_progress, "hidden")
            await self.call(self.ui.clear, [None])
            await self.call(self.ui.start, [None, *self.values])
            fresh = await self.call(self.ui.refresh, [None])
            self.assertIsInstance(fresh["data"][3], list)
        asyncio.run(scenario())

    def test_spectator_and_moderator_configuration(self):
        values = list(self.values)
        values[1] = False
        values[5] = True
        values[6] = "moderator-model"
        config = make_config(*values)
        self.assertTrue(all(p.backend.backend_type == "openai-chat" for p in config.players))
        self.assertEqual(config.environment.discussion_moderator.backend.model, "moderator-model")
        self.assertNotIn("terminal_condition", config.environment.discussion_moderator)
        self.assertNotIn("parallel", config.environment)

    def test_moderator_guide_prefilled_editable_and_empty_fallback(self):
        self.assertEqual(self.ui.settings[7].value, MODERATOR_GUIDE)
        values = list(self.values)
        values[5] = True
        self.assertEqual(make_config(*values).environment.discussion_moderator.role_desc,
                         MODERATOR_GUIDE)
        values[7] = "충분한 반론 기회를 주세요."
        self.assertEqual(make_config(*values).environment.discussion_moderator.role_desc,
                         values[7])
        values[7] = "   "
        self.assertEqual(make_config(*values).environment.discussion_moderator.role_desc,
                         MODERATOR_GUIDE)

    def test_spectator_can_see_private_messages_but_cannot_send(self):
        values = list(self.values)
        values[1] = False
        result = self.ui.start(None, *values)
        key = result[self.ui.key]
        self.ui.pause(key)
        rendered = self.ui.refresh(key)
        self.assertIn("전체 관전", rendered[self.ui.identity])
        self.assertIn("당신의 비밀 역할은 **경찰**", str(rendered[self.ui.chat_payload]))
        self.assertFalse(rendered[self.ui.send_button]["interactive"])
        self.assertIn("전송하지 못했습니다", self.ui.send(key, "hello")[self.ui.feedback])

    def test_roles_in_spectator_labels_only(self):
        for participate in (False, True):
            with self.subTest(participate=participate):
                values = list(self.values)
                values[1] = participate
                result = self.ui.start(None, *values)
                key = result[self.ui.key]
                self.ui.pause(key)
                session = self.ui.registry.get(key)
                with session.lock:
                    session.arena.environment.message_pool.append_message(
                        Message("Player 3", "의견을 듣고 싶습니다.", 0))
                    rendered = self.ui.render(session)
                label = "Player 3 (경찰)"
                for value in (rendered[self.ui.roster], str(rendered[self.ui.chat_payload])):
                    if participate:
                        self.assertNotIn(label, value)
                    else:
                        self.assertIn(label, value)
                self.ui.clear(key)
