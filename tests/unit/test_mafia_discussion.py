import threading
import time
import unittest

from chatarena.environments.mafia import Mafia
from chatarena.mafia_discussion import MafiaDiscussionController, control_token
from chatarena.message import Message
from chatarena.backends.openai import OpenAIChat

NAMES = ["A", "B", "C", "D"]
ROLES = dict(zip(NAMES, ["mafia", "doctor", "police", "villager"]))


def discussion_env(**kwargs):
    env = Mafia(NAMES, role_mapping=ROLES, **kwargs)
    env.step("A", "eliminate D")
    env.step("B", "protect D")
    env.step("C", "investigate A")
    assert env.phase == "DAY_DISCUSSION"
    return env


def drive(controller, condition, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        controller.tick()
        if condition():
            return
        time.sleep(.002)
    raise AssertionError("Timed out: " + controller.status)


class TestDiscussion(unittest.TestCase):
    def controller(self, env, query, **kwargs):
        ctrl = MafiaDiscussionController(env, query, **kwargs)
        self.addCleanup(ctrl.close)
        return ctrl

    def test_daytime_police_speech_does_not_investigate(self):
        env = discussion_env()
        before = list(env.get_observation())
        alive = set(env.alive_players)
        # Even explicit action-like speech is only dialogue during the day.
        env.step("C", "D를 조사하겠습니다.")
        env.discussion_speak("C", "D를 조사해 보니 마피아였습니다.")
        added = env.get_observation()[len(before):]
        self.assertEqual(len(added), 2)
        self.assertTrue(all(m.agent_name == "C" for m in added))
        self.assertEqual(env.phase, "DAY_DISCUSSION")
        self.assertEqual(env.alive_players, alive)
        self.assertEqual(env.discussion_messages, 2)
        self.assertFalse(any(m.agent_name == "Moderator" for m in added))

    def test_private_intents_and_zero_score_not_selected(self):
        env = discussion_env(max_discussion_messages=1)
        seen = []
        def query(req):
            seen.append(req)
            if req.kind == "intent":
                return "3" if req.player_name == "B" else "0"
            return "Why did A change their story?"
        ctrl = self.controller(env, query)
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        intents = [r for r in ctrl.records if r.request.kind == "intent"]
        self.assertEqual(len(intents), 4)
        self.assertTrue(all(r.trainable for r in intents))
        self.assertEqual([r.request.player_name for r in intents if r.selected], ["B"])
        self.assertEqual(env.last_speaker, "B")
        for req in seen:
            self.assertTrue(all(m.visible_to == "all" or req.player_name in m.visible_to
                                for m in req.observation))
        public = [m.content for m in env.get_observation() if m.visible_to == "all"]
        self.assertNotIn("3", public)
        self.assertNotIn("0", public)
        self.assertEqual(env.discussion_end_reason, "message_limit")

    def test_pass_is_private_and_reuses_scores(self):
        env = discussion_env()
        ctrl = self.controller(env, lambda req: "2" if req.kind == "intent" else "PASS")
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        self.assertEqual(sum(r.request.kind == "intent" for r in ctrl.records), 4)
        self.assertEqual(sum(r.request.kind == "speech" for r in ctrl.records), 4)
        self.assertEqual(env.discussion_messages, 0)
        self.assertFalse(any(m.content == "PASS" for m in env.get_observation()))
        self.assertTrue(all(r.trainable for r in ctrl.records))

    def test_weighting_and_seed(self):
        selected = []
        for _ in range(2):
            env = discussion_env(seed=91, max_discussion_messages=1)
            env.last_speaker = "B"
            ctrl = self.controller(env, lambda req: "2" if req.kind == "intent" else "hello")
            ctrl._scores = {"A": 0, "B": 3, "C": 2, "D": 1}
            self.assertEqual(ctrl.selection_weights(), {"B": 4.5, "C": 4, "D": 1})
            drive(ctrl, lambda: env.phase == "DAY_VOTING")
            selected.append(env.last_speaker)
        self.assertEqual(*selected)

    def test_formatted_control_replies_stay_private(self):
        for reply in ["PASS.", '"PASS"', "**pass**", "I'll listen for now. PASS."]:
            with self.subTest(reply=reply):
                env = discussion_env()
                ctrl = self.controller(env, lambda req: "2" if req.kind == "intent" else reply)
                drive(ctrl, lambda: env.phase == "DAY_VOTING")
                self.assertEqual(env.discussion_messages, 0)
                self.assertFalse(any(m.content == reply for m in env.get_observation()))
                speech = [r for r in ctrl.records if r.request.kind == "speech"]
                self.assertTrue(all(r.response == reply and not r.trainable for r in speech))
                self.assertEqual(sum(d["passes"] for d in ctrl.diagnostics().values()), 4)

    def test_punctuated_moderator_commands_stay_private(self):
        for reply in ["CONTINUE.", "continue", "`END_DISCUSSION`"]:
            with self.subTest(reply=reply):
                env = discussion_env()
                ctrl = self.controller(env, lambda req: "0", moderator_query=lambda req: reply)
                drive(ctrl, lambda: env.phase == "DAY_VOTING")
                self.assertFalse(any(m.content == reply for m in env.get_observation()))
                self.assertEqual(sum(r.request.kind == "moderation" for r in ctrl.records), 1)

    def test_wrong_role_control_is_error_not_public_speech(self):
        env = discussion_env()
        ctrl = self.controller(env, lambda req: "2" if req.kind == "intent" else "CONTINUE.")
        drive(ctrl, lambda: ctrl.status == "error")
        self.assertEqual(env.discussion_messages, 0)
        self.assertFalse(ctrl.records[-1].trainable)

    def test_ordinary_mentions_are_not_controls(self):
        for text in ["Please continue explaining.", "Why did A say PASS?", "I suspect A. PASS."]:
            self.assertIsNone(control_token(text))

    def test_silence_moderator_once_and_public_only(self):
        env = discussion_env()
        requests = []
        def moderator(req):
            requests.append(req)
            return "Any final thoughts?"
        ctrl = self.controller(env, lambda req: "0", moderator_query=moderator)
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        self.assertEqual(len(requests), 1)
        self.assertTrue(all(m.visible_to == "all" for m in requests[0].observation))
        self.assertEqual(sum(r.request.kind == "intent" for r in ctrl.records), 4)
        self.assertFalse(any(m.content == "Any final thoughts?" for m in env.get_observation()))
        self.assertFalse(any(r.trainable for r in ctrl.records if r.request.kind == "moderation"))

    def test_moderator_cannot_call_on_players_or_publish_guidance(self):
        env = discussion_env(max_discussion_messages=3)
        announcements = ['A님 먼저 말해주세요.', '다른 의견을 더 들어봅시다.']
        moderator_calls = []
        def moderator(req):
            moderator_calls.append(req)
            return announcements[len(moderator_calls) - 1]
        ctrl = self.controller(env, lambda req: '2' if req.kind == 'intent' else '새 근거입니다.',
                               moderator_query=moderator)
        drive(ctrl, lambda: env.phase == 'DAY_VOTING')
        self.assertEqual(env.discussion_messages, 3)
        self.assertEqual(len(moderator_calls), 2)
        self.assertFalse(any(m.content in announcements for m in env.get_observation()))
        records = [r for r in ctrl.records if r.request.kind == 'moderation']
        self.assertTrue(all(not r.valid and not r.trainable for r in records))
        self.assertEqual(env.discussion_end_reason, 'message_limit')

    def test_moderator_can_end_after_speech(self):
        env = discussion_env()
        ctrl = self.controller(env, lambda req: "1" if req.kind == "intent" else "hello",
                               moderator_query=lambda req: "END_DISCUSSION")
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        self.assertEqual(env.discussion_messages, 1)
        self.assertEqual(env.discussion_end_reason, "moderator")

    def test_invalid_retry_excluded_from_training(self):
        env = discussion_env()
        def query(req):
            if req.kind == "intent":
                return "0" if req.retry else "I choose 3"
            return "PASS"
        ctrl = self.controller(env, query)
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        self.assertEqual(len(ctrl.records), 8)
        self.assertFalse(any(r.trainable for r in ctrl.records))
        self.assertEqual(sum(r.valid for r in ctrl.records), 4)

    def test_failure_is_not_silence(self):
        env = discussion_env()
        ctrl = self.controller(env, lambda req: "invalid")
        drive(ctrl, lambda: ctrl.status == "error")
        self.assertEqual(env.phase, "DAY_DISCUSSION")
        self.assertIn("All AI", ctrl.error)

    def test_human_interrupts_inflight_speech(self):
        env = discussion_env()
        entered, release = threading.Event(), threading.Event()
        def query(req):
            if req.kind == "intent":
                return "3" if req.player_name == "A" else "0"
            entered.set()
            release.wait(2)
            return "obsolete speech"
        ctrl = self.controller(env, query, human_names=["D"])
        self.addCleanup(release.set)
        drive(ctrl, entered.is_set)
        ctrl.submit_human("D", "Wait, answer my question first.")
        self.assertEqual(env.last_speaker, "D")
        release.set()
        drive(ctrl, lambda: any(r.stale and r.request.kind == "speech" for r in ctrl.records))
        ctrl.pause()
        stale = [r for r in ctrl.records if r.stale]
        self.assertFalse(any(r.trainable for r in stale))
        self.assertFalse(any(m.content == "obsolete speech" for m in env.get_observation()))

    def test_reset_discards_pending_results(self):
        env = discussion_env()
        release = threading.Event()
        ctrl = self.controller(env, lambda req: (release.wait(2) and "3") or "0")
        self.addCleanup(release.set)
        ctrl.tick()
        old_session = env.session_id
        ctrl.reset()
        ctrl.pause()
        release.set()
        drive(ctrl, lambda: not ctrl.pending)
        self.assertNotEqual(old_session, env.session_id)
        self.assertTrue(all(r.stale and not r.trainable for r in ctrl.records))
        self.assertEqual(env.phase, "NIGHT_MAFIA")

    def test_clock_excludes_pause_and_waits_for_human(self):
        now = [0.0]
        env = discussion_env(discussion_seconds=20, silence_seconds=10)
        ctrl = self.controller(env, lambda req: "0", human_names=["D"],
                               realtime=True, clock=lambda: now[0])
        drive(ctrl, lambda: ctrl.status == "waiting_human")
        now[0] = 5
        ctrl.pause()
        now[0] = 100
        ctrl.resume()
        ctrl.tick()
        self.assertEqual(env.phase, "DAY_DISCUSSION")
        self.assertEqual(ctrl._elapsed, 5)
        now[0] = 116
        ctrl.tick()
        self.assertEqual(env.discussion_end_reason, "time_limit")

    def test_round_limit_and_moderator_does_not_self_recurse(self):
        env = discussion_env(max_intent_rounds=1)
        ctrl = self.controller(env, lambda req: "2" if req.kind == "intent" else "hello",
                               moderator_query=lambda req: "Please continue.")
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        self.assertEqual(env.discussion_end_reason, "intent_limit")
        self.assertEqual(sum(r.request.kind == "moderation" for r in ctrl.records), 1)

    def test_dead_human_and_wrong_night_turn_rejected(self):
        env = discussion_env()
        ctrl = self.controller(env, lambda req: "0", human_names=["D"])
        env.alive_players.remove("D")
        with self.assertRaises(ValueError):
            ctrl.submit_human("D", "hello")
        env.reset()
        with self.assertRaises(ValueError):
            ctrl.submit_human("D", "kill A")

    def test_intents_are_parallel_but_from_one_snapshot(self):
        env = discussion_env()
        barrier = threading.Barrier(4)
        versions = []
        def query(req):
            versions.append(req.version)
            barrier.wait(timeout=2)
            return "0"
        ctrl = self.controller(env, query)
        drive(ctrl, lambda: env.phase == "DAY_VOTING")
        self.assertEqual(len(versions), 4)
        self.assertEqual(len(set(versions)), 1)

    def test_timeout_is_error_and_old_work_is_bounded(self):
        now = [0.0]
        release = threading.Event()
        env = discussion_env()
        ctrl = self.controller(env, lambda req: (release.wait(2) and "0") or "0",
                               clock=lambda: now[0], request_timeout=1)
        self.addCleanup(release.set)
        ctrl.tick()
        self.assertEqual(len(ctrl.pending), 4)
        now[0] = 2
        ctrl.tick()
        self.assertEqual(ctrl.status, "error")
        for _ in range(5):
            ctrl.resume()
            ctrl.tick()
        self.assertEqual(len(ctrl.pending), 4)
        self.assertEqual(env.phase, "DAY_DISCUSSION")
        release.set()
        drive(ctrl, lambda: not ctrl.pending)
        self.assertTrue(all(r.stale and not r.trainable for r in ctrl.records))

    def test_simultaneous_human_submissions_commit_once_each(self):
        env = discussion_env()
        ctrl = self.controller(env, lambda req: "0", human_names=["D"])
        threads = [threading.Thread(target=ctrl.submit_human, args=("D", text))
                   for text in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        messages = [m for m in env.get_observation() if m.agent_name == "D"]
        self.assertEqual({m.content for m in messages}, {"first", "second"})
        self.assertEqual(len(messages), 2)
        self.assertNotEqual(messages[0].turn, messages[1].turn)

    def test_formatter_does_not_merge_other_speaker_into_assistant(self):
        messages = [Message("A", "my speech", 0), Message("B", "reply", 1)]
        formatted = OpenAIChat.format_messages("A", "rules", messages,
                                               Message("System", "score only", -1))
        self.assertEqual([m["role"] for m in formatted], ["system", "assistant", "user", "system"])
        self.assertNotIn("reply", formatted[1]["content"])
        self.assertNotEqual(formatted[1]["name"], formatted[2]["name"])
        self.assertEqual(formatted[-1]["content"], "score only")


if __name__ == "__main__":
    unittest.main()
