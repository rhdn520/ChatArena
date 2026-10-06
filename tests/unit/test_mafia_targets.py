import unittest

from chatarena.environments.mafia import Mafia


class TestMafiaTargets(unittest.TestCase):
    def setUp(self):
        self.env = Mafia(['Player ' + str(i) for i in range(1, 11)], seed=7)

    def test_number_boundaries_and_alias_positions(self):
        for candidates in (['Player 1', 'Player 2', 'Player 10'],
                           ['Player 10', 'Player 2', 'Player 1']):
            for reply, expected in [('Player 10에게 투표합니다.', 'Player 10'),
                                    ('Player1 대신 Player2에게 투표합니다.', 'Player 2'),
                                    ('Player_2 대신 player_10을 조사합니다.', 'Player 10'),
                                    ('I vote to eliminate [Player 1].', 'Player 1')]:
                with self.subTest(reply=reply, candidates=candidates):
                    self.assertEqual(self.env._parse_target(reply, candidates), expected)
        self.assertIsNone(self.env._parse_target('Player 10', ['Player 1']))
        self.assertIsNone(self.env._parse_target('SomeonePlayer1', ['Player 1']))
        self.assertIsNone(self.env._parse_target('아무도 모르겠습니다', ['Player 1']))

    def test_no_target_does_not_mutate_or_pick_random_player(self):
        env = Mafia(list('ABCD'), role_mapping={'A': 'mafia', 'B': 'doctor', 'C': 'police', 'D': 'villager'})
        before = (env.version, len(env.get_observation()), env.get_next_player())
        with self.assertRaises(ValueError):
            env.step('A', '선택하지 못했습니다.')
        self.assertEqual((env.version, len(env.get_observation()), env.get_next_player()), before)
        self.assertEqual(env.night_kills, [])

    def test_seeded_presentation_and_voting_order(self):
        def orders():
            env = Mafia(['Player ' + str(i) for i in range(1, 11)], seed=7)
            samples = [env.randomized_names(env.alive_players) for _ in range(8)]
            env._start_voting_phase()
            return samples, [env.get_next_player(), *env._action_queue], env.player_roles
        first, second = orders(), orders()
        self.assertEqual(first, second)
        self.assertGreater(len({order[0] for order in first[0]}), 1)
        self.assertEqual(set(first[1]), set(self.env.player_names))
        # Presentation does not consume gameplay RNG or alter role assignment.
        self.assertEqual(first[2], self.env.player_roles)

    def test_invalid_ai_action_is_not_published_or_learned(self):
        from chatarena.mafia_discussion import MafiaDiscussionController
        from test_mafia_discussion import drive
        env = Mafia(list('ABCD'), role_mapping={'A': 'mafia', 'B': 'doctor', 'C': 'police', 'D': 'villager'})
        ctrl = MafiaDiscussionController(env, lambda req: '대상을 고르지 못했습니다.')
        self.addCleanup(ctrl.close)
        before = len(env.get_observation())
        drive(ctrl, lambda: ctrl.status == 'error')
        self.assertEqual(len(env.get_observation()), before)
        self.assertEqual(env.get_next_player(), 'A')
        self.assertFalse(ctrl.records[-1].valid)
        self.assertFalse(ctrl.records[-1].trainable)
