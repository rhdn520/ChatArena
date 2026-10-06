import unittest

from chatarena.rl.mafia_rollout import MafiaRolloutManager, MafiaTurnRecord, compute_mafia_reward

NAMES = ["A", "B", "C", "D"]
ROLES = dict(zip(NAMES, ["mafia", "doctor", "police", "villager"]))


def policy(name, prompt):
    if "REQUEST intent:" in prompt:
        return "0"
    if "NIGHT_MAFIA입니다" in prompt:
        return "eliminate D"
    if "NIGHT_DOCTOR입니다" in prompt:
        return "protect D"
    if "NIGHT_POLICE입니다" in prompt:
        return "investigate A"
    return "vote A"


class TestRollout(unittest.TestCase):
    def test_zero_and_unselected_intents_are_learned(self):
        manager = MafiaRolloutManager(player_names=NAMES, role_mapping=ROLES)
        result = manager.rollout_episode("A", policy, policy)
        self.assertFalse(result.truncated)
        self.assertEqual(result.winner, "citizens")
        for name, traj in result.trajectories.items():
            intent = next(t for t in traj.turns if t.kind == "intent")
            self.assertEqual(intent.score, 0)
            self.assertFalse(intent.selected)
            self.assertTrue(intent.trainable)
            self.assertEqual(traj.shaped_reward, traj.raw_reward)
        self.assertEqual(result.discussion_endings[0]["reason"], "silence")
        self.assertEqual(result.diagnostics["A"]["intent_counts"]["0"], 1)

    def test_truncated_not_draw_and_no_training(self):
        result = MafiaRolloutManager(player_names=NAMES, role_mapping=ROLES,
                                    max_total_steps=1).rollout_episode("A", policy, policy)
        self.assertTrue(result.truncated)
        self.assertEqual(result.winner, "truncated")
        self.assertEqual(result.termination_reason, "step_limit")
        self.assertFalse(any(t.trainable for tr in result.trajectories.values() for t in tr.turns))

    def test_model_failure_truncates(self):
        def bad(name, prompt):
            return "bad" if "REQUEST intent:" in prompt else policy(name, prompt)
        result = MafiaRolloutManager(player_names=NAMES, role_mapping=ROLES).rollout_episode("A", bad, bad)
        self.assertTrue(result.truncated)
        self.assertTrue(result.termination_reason.startswith("model_error"))

    def test_true_draw(self):
        def draw(name, prompt):
            if "DAY_VOTING입니다" in prompt:
                return "vote " + name
            return policy(name, prompt)
        result = MafiaRolloutManager(player_names=NAMES, role_mapping=ROLES,
                                    max_days=1).rollout_episode("A", draw, draw)
        self.assertFalse(result.truncated)
        self.assertEqual(result.winner, "draw")
        self.assertEqual(set(result.rewards.values()), {0.0})

    def test_reward_does_not_pay_for_talking(self):
        turns = [MafiaTurnRecord(0, "DAY_DISCUSSION", "p", "hello") for _ in range(30)]
        self.assertEqual(compute_mafia_reward("A", "mafia", -1, False, turns), -1)
