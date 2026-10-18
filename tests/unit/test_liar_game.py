import json
import os
import tempfile
import unittest

from chatarena.arena import Arena
from chatarena.config import ArenaConfig
from chatarena.environments import load_environment
from chatarena.environments.liar_game import LiarGame
from chatarena.environments.liar_game_words import DEFAULT_WORD_SETS


class TestLiarGameEnvironment(unittest.TestCase):
    def setUp(self):
        self.players = ["Player 1", "Player 2", "Player 3"]

    def make_env(self, rounds=1, reveal_topic=True):
        return LiarGame(
            player_names=self.players,
            clue_rounds=rounds,
            reveal_topic=reveal_topic,
            word_sets={"Food & Drinks": ["Pizza", "Apple"], "Brands": ["Apple"]},
            liar_name="Player 3",
            topic="Food & Drinks",
            secret_word="Pizza",
        )

    def finish_clues(self, env):
        for _ in range(len(self.players) * env.clue_rounds):
            actor = env.get_next_player()
            self.assertEqual(env.phase, "clue")
            self.assertFalse(env.step(actor, "Often shared at a party.").terminal)
        self.assertEqual(env.phase, "vote")

    def vote(self, env, targets):
        timestep = None
        for voter, target in zip(self.players, targets):
            self.assertEqual(env.get_next_player(), voter)
            timestep = env.step(voter, f"I vote for {target}")
        return timestep

    @staticmethod
    def contents(messages):
        return "\n".join(message.content for message in messages)

    def test_dataset_has_ten_topics_and_one_thousand_entries(self):
        self.assertEqual(
            list(DEFAULT_WORD_SETS),
            [
                "Food & Drinks", "Animals", "Places", "Occupations",
                "Everyday Objects", "Nature & Environment", "Movies",
                "Countries", "Cities", "Brands",
            ],
        )
        self.assertEqual(len(DEFAULT_WORD_SETS), 10)
        self.assertTrue(all(len(words) == 100 for words in DEFAULT_WORD_SETS.values()))
        self.assertEqual(sum(map(len, DEFAULT_WORD_SETS.values())), 1000)
        self.assertIn("Apple", DEFAULT_WORD_SETS["Food & Drinks"])
        self.assertIn("Apple", DEFAULT_WORD_SETS["Brands"])
        self.assertIn("Singapore", DEFAULT_WORD_SETS["Countries"])
        self.assertIn("Singapore", DEFAULT_WORD_SETS["Cities"])

    def test_sampling_assigns_one_liar_and_valid_topic_word(self):
        config = {
            "env_type": "liar_game",
            "player_names": self.players,
            "random_seed": 73,
        }
        first = load_environment(dict(config))
        second = load_environment(dict(config))
        self.assertIsInstance(first, LiarGame)
        self.assertEqual(first.liar_name, second.liar_name)
        self.assertEqual(first.topic, second.topic)
        self.assertEqual(first.secret_word, second.secret_word)
        self.assertEqual(len(first.non_liar_names), 2)
        self.assertNotIn(first.liar_name, first.non_liar_names)
        self.assertIn(first.topic, DEFAULT_WORD_SETS)
        self.assertIn(first.secret_word, DEFAULT_WORD_SETS[first.topic])

    def test_example_configs_use_registered_liar_game(self):
        for path in (
            "examples/liar_game.json",
            "examples/liar_game_qwen3_5_9b.json",
        ):
            with self.subTest(path=path):
                config = ArenaConfig.load(path)
                self.assertEqual(config.environment["env_type"], "liar_game")
                self.assertEqual(config.environment["clue_rounds"], 2)
                self.assertTrue(config.environment["reveal_topic"])
                self.assertEqual(len(config.players), 5)
        qwen_config = ArenaConfig.load("examples/liar_game_qwen3_5_9b.json")
        self.assertEqual(
            {player["backend"]["model"] for player in qwen_config.players},
            {"Qwen/Qwen3.5-9B"},
        )

    def test_revealed_topic_is_visible_to_everyone_but_word_only_to_non_liars(self):
        env = self.make_env(reveal_topic=True)
        for player in self.players:
            observation = self.contents(env.get_observation(player))
            self.assertIn("Topic: Food & Drinks", observation)
            if player == env.liar_name:
                self.assertNotIn("Pizza", observation)
                self.assertIn("You are the Liar", observation)
            else:
                self.assertIn("Pizza", observation)
        public_messages = [
            msg for msg in env.get_observation() if msg.visible_to == "all"
        ]
        self.assertNotIn("Pizza", self.contents(public_messages))

    def test_hidden_topic_is_absent_from_all_preterminal_observations(self):
        env = self.make_env(reveal_topic=False)
        for player in self.players:
            observation = self.contents(env.get_observation(player))
            self.assertNotIn("Food & Drinks", observation)
            self.assertEqual("Pizza" in observation, player != env.liar_name)
        self.assertNotIn(
            "Food & Drinks",
            self.contents(
                msg for msg in env.get_observation() if msg.agent_name == "Moderator"
            ),
        )

    def test_clue_turns_follow_player_order_and_round_count(self):
        env = self.make_env(rounds=2)
        for round_number in (1, 2):
            for player in self.players:
                self.assertEqual(env.phase, "clue")
                self.assertEqual(env.current_round, round_number)
                self.assertEqual(env.get_next_player(), player)
                prompt = env.get_observation(player)[-1].content
                self.assertIn(f"Round {round_number}/2: {player}", prompt)
                self.assertNotIn("Player 3 is the Liar", prompt)
                env.step(player, f"Clue {round_number} from {player}")
        self.assertEqual(env.phase, "vote")
        self.assertEqual(env.current_round, 3)
        self.assertEqual(env.get_next_player(), "Player 1")
        self.assertEqual(
            sum(msg.agent_name in self.players for msg in env.get_observation()), 6
        )

    def test_liar_never_sees_word_during_clues_and_voting(self):
        for reveal_topic in (True, False):
            with self.subTest(reveal_topic=reveal_topic):
                env = self.make_env(rounds=2, reveal_topic=reveal_topic)

                def check_hidden():
                    observation = self.contents(env.get_observation(env.liar_name))
                    self.assertNotIn("Pizza", observation)
                    if not reveal_topic:
                        self.assertNotIn("Food & Drinks", observation)

                check_hidden()
                for _ in range(len(self.players) * env.clue_rounds):
                    env.step(env.get_next_player(), "Often eaten with friends.")
                    check_hidden()
                env.step("Player 1", "I vote for Player 3")
                check_hidden()
                env.step("Player 2", "I vote for Player 3")
                check_hidden()
                env.step("Player 3", "I vote for Player 1")
                self.assertEqual(env.phase, "liar_guess")
                check_hidden()

    def test_wrong_accusation_gives_liar_win_and_binary_rewards(self):
        env = self.make_env()
        self.finish_clues(env)
        timestep = self.vote(env, ["Player 2", "Player 1", "Player 2"])
        self.assertTrue(timestep.terminal)
        self.assertEqual(env.accused_player, "Player 2")
        self.assertEqual(env.winner, "liar")
        self.assertIsNone(env.word_guess)
        self.assertEqual(
            timestep.reward,
            {"Player 1": 0.0, "Player 2": 0.0, "Player 3": 1.0},
        )

    def test_ballots_are_private_until_terminal_result(self):
        env = self.make_env()
        self.finish_clues(env)
        for player in self.players:
            own = env.get_observation(player)
            self.assertTrue(any(
                message.msg_type == "instruction" and message.visible_to == [player]
                for message in own
            ))
            self.assertFalse(any(
                message.msg_type == "instruction" and message.visible_to == [other]
                for other in self.players if other != player for message in own
            ))
        env.step("Player 1", "I vote for Player 2")
        self.assertNotIn("I vote for Player 2", self.contents(env.get_observation("Player 2")))
        self.assertNotIn("Votes:", self.contents(env.get_observation("Player 2")))
        env.step("Player 2", "I vote for Player 1")
        self.assertNotIn("I vote for Player 1", self.contents(env.get_observation("Player 1")))
        env.step("Player 3", "I vote for Player 2")
        self.assertTrue(env.is_terminal())
        self.assertIn("Votes: Player 1 -> Player 2", self.contents(
            env.get_observation("Player 1")
        ))

    def test_caught_liar_correct_guess_gives_liar_win(self):
        env = self.make_env()
        self.finish_clues(env)
        timestep = self.vote(env, ["Player 3", "Player 3", "Player 1"])
        self.assertFalse(timestep.terminal)
        self.assertEqual(env.phase, "liar_guess")
        self.assertNotIn("Pizza", self.contents(env.get_observation("Player 3")))
        timestep = env.step("Player 3", "I guess the word is 'pIzZa'.")
        self.assertTrue(timestep.terminal)
        self.assertEqual(env.winner, "liar")
        self.assertEqual(
            timestep.reward,
            {"Player 1": 0.0, "Player 2": 0.0, "Player 3": 1.0},
        )

    def test_caught_liar_wrong_guess_gives_non_liars_win(self):
        env = self.make_env()
        self.finish_clues(env)
        self.vote(env, ["Player 3", "Player 3", "Player 1"])
        timestep = env.step("Player 3", "My final guess is Apple.")
        self.assertTrue(timestep.terminal)
        self.assertEqual(env.winner, "non_liars")
        self.assertEqual(
            timestep.reward,
            {"Player 1": 1.0, "Player 2": 1.0, "Player 3": 0.0},
        )
        self.assertIn("Secret word: Pizza", self.contents(env.get_observation("Player 3")))

    def test_guess_does_not_match_inside_another_word(self):
        env = self.make_env()
        self.assertFalse(env._is_correct_word_guess("Pizzazz"))
        self.assertFalse(env._is_correct_word_guess("Pizza Hut"))
        self.assertTrue(env._is_correct_word_guess('The word is "pizza".'))
        self.assertTrue(env._is_correct_word_guess("I guess the word is Pizza."))

    def test_malformed_and_self_votes_abstain_and_tie_is_deterministic(self):
        env = self.make_env()
        self.finish_clues(env)
        env.step("Player 1", "I refuse to vote")
        env.step("Player 2", "I vote for Player 2")
        timestep = env.step("Player 3", "I vote for Player 2")
        self.assertTrue(timestep.terminal)
        self.assertIsNone(env.votes["Player 1"])
        self.assertIsNone(env.votes["Player 2"])
        self.assertEqual(env.accused_player, "Player 2")

        env.reset()
        self.finish_clues(env)
        env.step("Player 1", "abstain")
        env.step("Player 2", "I vote for Player 1")
        env.step("Player 3", "I vote for Player 2")
        self.assertEqual(env.accused_player, "Player 1")
        self.assertIn("A tie was resolved", self.contents(env.get_observation()))

    def test_reset_clears_episode_state_and_history(self):
        env = self.make_env()
        env.step("Player 1", "A clue from the old episode")
        env.votes["Player 1"] = "Player 2"
        env.word_guess = "Apple"
        old_conversation_id = env.message_pool.conversation_id
        timestep = env.reset()
        self.assertFalse(timestep.terminal)
        self.assertEqual(env.phase, "clue")
        self.assertEqual(env.current_round, 1)
        self.assertEqual(env.votes, {})
        self.assertIsNone(env.word_guess)
        self.assertIsNone(env.winner)
        self.assertEqual(env.get_rewards(), env.get_zero_rewards())
        self.assertNotIn("old episode", self.contents(env.get_observation()))
        self.assertEqual(env.message_pool.conversation_id, old_conversation_id)

    def test_history_contains_clues_votes_guess_and_terminal_metadata(self):
        env = self.make_env(reveal_topic=False)
        arena = Arena([], env)
        self.finish_clues(env)
        self.vote(env, ["Player 3", "Player 3", "Player 1"])
        env.step("Player 3", "Apple")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "history.json")
            arena.save_history(path)
            with open(path, encoding="utf-8") as history_file:
                history = json.load(history_file)
        text = self.contents(env.get_observation())
        self.assertEqual(len(history), len(env.get_observation()))
        self.assertIn("Topic reveal: off", text)
        self.assertIn("Topic: Food & Drinks", text)
        self.assertIn("Secret word: Pizza", text)
        self.assertIn("Actual Liar: Player 3", text)
        self.assertIn("Final guess: Apple", text)
        self.assertIn("Winner: Non-liars", text)
        self.assertIn("Round 1/1", text)
        self.assertIn("Votes:", text)


if __name__ == "__main__":
    unittest.main()
