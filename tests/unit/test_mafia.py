import unittest
from chatarena.environments import load_environment
from chatarena.environments.mafia import (
    Mafia,
    MAFIA,
    DOCTOR,
    POLICE,
    VILLAGER,
)


class TestMafiaEnvironment(unittest.TestCase):
    def setUp(self):
        self.player_names = ["Player 1", "Player 2", "Player 3", "Player 4"]
        self.role_mapping = {
            "Player 1": MAFIA,
            "Player 2": DOCTOR,
            "Player 3": POLICE,
            "Player 4": VILLAGER,
        }

    def test_initialization_and_roles(self):
        env = Mafia(
            player_names=self.player_names,
            role_mapping=self.role_mapping,
            max_days=3,
        )
        self.assertEqual(env.player_roles["Player 1"], MAFIA)
        self.assertEqual(env.player_roles["Player 2"], DOCTOR)
        self.assertEqual(env.player_roles["Player 3"], POLICE)
        self.assertEqual(env.player_roles["Player 4"], VILLAGER)
        self.assertEqual(len(env.alive_players), 4)

        # Check information visibility
        obs_p1 = env.get_observation("Player 1")
        obs_p4 = env.get_observation("Player 4")

        # Player 1 should see secret role as MAFIA
        self.assertTrue(any("당신의 비밀 역할은 **마피아**" in m.content for m in obs_p1))
        # Player 4 (Villager) should NOT see who the MAFIA is
        self.assertFalse(any("당신의 비밀 역할은 **마피아**" in m.content for m in obs_p4))

    def test_load_environment_from_config(self):
        config = {
            "env_type": "mafia",
            "player_names": self.player_names,
            "role_mapping": self.role_mapping,
        }
        env = load_environment(config)
        self.assertIsInstance(env, Mafia)
        self.assertEqual(env.player_roles, self.role_mapping)

    def test_korean_actions_and_private_reports_through_victory(self):
        env = Mafia(self.player_names, role_mapping=self.role_mapping)
        env.step("Player 1", "Player 4를 제거하겠습니다.")
        self.assertEqual(env.night_kills, ["Player 4"])
        env.step("Player 2", "Player 4를 보호하겠습니다.")
        self.assertEqual(env.night_heal, "Player 4")
        env.step("Player 3", "Player 1을 조사하겠습니다.")
        report = "조사 결과: **Player 1** — **마피아**."
        self.assertTrue(any(m.content == report for m in env.get_observation("Player 3")))
        self.assertFalse(any(m.content == report for m in env.get_observation("Player 4")))
        self.assertEqual(len(env.alive_players), 4)
        env.discussion_speak("Player 3", "Player 1이 마피아라고 생각합니다.")
        env.end_discussion("test")
        for _ in range(4):
            env.step(env.get_next_player(), "Player 1에게 투표합니다.")
        self.assertEqual(env.termination_reason, "citizens_win")
        self.assertIn("시민 팀이 승리했습니다", env.get_observation()[-1].content)
        self.assertTrue(all(any("가" <= ch <= "힣" for ch in m.content)
                            for m in env.get_observation() if m.agent_name == "Moderator"))

    def test_game_flow_full_cycle(self):
        env = Mafia(
            player_names=self.player_names,
            role_mapping=self.role_mapping,
            max_days=3,
        )

        # 1. Night: Mafia turn (Player 1)
        self.assertEqual(env.phase, "NIGHT_MAFIA")
        self.assertEqual(env.get_next_player(), "Player 1")
        ts = env.step("Player 1", "I choose to eliminate Player 4")
        self.assertFalse(ts.terminal)

        # 2. Night: Doctor turn (Player 2)
        self.assertEqual(env.phase, "NIGHT_DOCTOR")
        self.assertEqual(env.get_next_player(), "Player 2")
        # Doctor protects Player 3 (fails to protect Player 4)
        ts = env.step("Player 2", "I choose to protect Player 3")
        self.assertFalse(ts.terminal)

        # 3. Night: Police turn (Player 3)
        self.assertEqual(env.phase, "NIGHT_POLICE")
        self.assertEqual(env.get_next_player(), "Player 3")
        ts = env.step("Player 3", "I investigate Player 1")
        self.assertFalse(ts.terminal)

        # Police should have received investigation report
        police_obs = env.get_observation("Player 3")
        self.assertTrue(
            any("조사 결과: **Player 1** — **마피아**" in m.content for m in police_obs)
        )

        # 4. Daybreak: Player 4 was eliminated
        self.assertNotIn("Player 4", env.alive_players)
        self.assertEqual(env.phase, "DAY_DISCUSSION")

        # Free discussion has no fixed speaker order; the controller ends it.
        env.step("Player 3", "I investigated Player 1.")
        env.step("Player 2", "Then I suspect Player 1.")
        self.assertEqual(env.phase, "DAY_DISCUSSION")
        env.end_discussion("test")

        # 6. Voting Phase
        self.assertEqual(env.phase, "DAY_VOTING")
        voter1 = env.get_next_player()
        env.step(voter1, "I vote to eliminate Player 1")

        voter2 = env.get_next_player()
        env.step(voter2, "I vote to eliminate Player 1")

        voter3 = env.get_next_player()
        ts = env.step(voter3, "I vote to eliminate Player 2")

        # Player 1 (Mafia) should be eliminated by 2 votes
        self.assertNotIn("Player 1", env.alive_players)
        # All Mafia eliminated -> Citizens win!
        self.assertTrue(ts.terminal)
        self.assertEqual(ts.reward["Player 1"], -1.0)
        self.assertEqual(ts.reward["Player 2"], 1.0)
        self.assertEqual(ts.reward["Player 3"], 1.0)
        self.assertEqual(ts.reward["Player 4"], 1.0)

    def test_doctor_saves_victim(self):
        env = Mafia(
            player_names=self.player_names,
            role_mapping=self.role_mapping,
            max_days=3,
        )

        # Mafia targets Player 4
        env.step("Player 1", "I choose to eliminate Player 4")
        # Doctor successfully protects Player 4
        env.step("Player 2", "I choose to protect Player 4")
        # Police investigates Player 2
        env.step("Player 3", "I choose to investigate Player 2")

        # Player 4 should still be alive!
        self.assertIn("Player 4", env.alive_players)
        self.assertEqual(len(env.alive_players), 4)

    def test_mafia_wins_by_numbers(self):
        # 3 player game: 1 Mafia, 2 Citizens
        role_map = {
            "Player 1": MAFIA,
            "Player 2": DOCTOR,
            "Player 3": VILLAGER,
        }
        env = Mafia(
            player_names=["Player 1", "Player 2", "Player 3"],
            role_mapping=role_map,
            max_days=3,
        )

        # Night: Mafia kills Player 3
        env.step("Player 1", "I choose to eliminate Player 3")
        # Doctor protects self
        env.step("Player 2", "I choose to protect Player 2")

        # Day breaks: Player 3 dies. Living: Player 1 (Mafia), Player 2 (Citizen)
        # Alive Mafia (1) >= Alive Citizens (1) -> Mafia wins immediately!
        self.assertTrue(env.is_terminal())
        rewards = env.get_rewards()
        self.assertEqual(rewards["Player 1"], 1.0)
        self.assertEqual(rewards["Player 2"], -1.0)
        self.assertEqual(rewards["Player 3"], -1.0)

    def test_rollout_manager(self):
        from chatarena.rl.mafia_rollout import MafiaRolloutManager

        manager = MafiaRolloutManager(
            player_names=["Player 1", "Player 2", "Player 3", "Player 4"],
            role_mapping={
                "Player 1": MAFIA,
                "Player 2": DOCTOR,
                "Player 3": POLICE,
                "Player 4": VILLAGER,
            },
            max_days=2,
        )

        def mock_policy(player, prompt):
            if "REQUEST intent:" in prompt:
                return "0"
            if "NIGHT_MAFIA입니다" in prompt:
                return "I choose to eliminate Player 4"
            elif "NIGHT_DOCTOR입니다" in prompt:
                return "I choose to protect Player 2"
            elif "NIGHT_POLICE입니다" in prompt:
                return "I choose to investigate Player 1"
            elif "DAY_VOTING입니다" in prompt:
                return "I vote to eliminate Player 1"
            return "I suspect Player 1"

        result = manager.rollout_episode(
            policy_agent_names=["Player 1"],
            policy_fn=mock_policy,
            opponent_fn=mock_policy,
        )

        self.assertIn(result.winner, ["mafia", "citizens", "draw"])
        self.assertIn("Player 1", result.trajectories)
        self.assertTrue(len(result.trajectories["Player 1"].turns) > 0)
        self.assertIsInstance(result.trajectories["Player 1"].shaped_reward, float)


if __name__ == "__main__":
    unittest.main()
