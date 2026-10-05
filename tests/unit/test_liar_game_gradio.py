import unittest
from pathlib import Path
from unittest.mock import patch

from chatarena.arena import Arena
from chatarena.environments.liar_game import LiarGame
from chatarena.environments.base import TimeStep
from chatarena.ui.liar_game_gradio import (
    HumanLiarGame,
    LiarGameRoom,
    RoomRegistry,
    create_human_game,
    render_snapshot,
    validate_clue_rounds,
    validate_nickname,
    validate_player_counts,
)


NAMES = [f"Player {number}" for number in range(1, 6)]


class FakePlayer:
    def __init__(self, name, environment, human=False, vote_target="Player 5"):
        self.name = name
        self.environment = environment
        self.human = human
        self.vote_target = vote_target

    def __call__(self, observation):
        if self.human:
            raise AssertionError("The UI must not invoke the human backend")
        if self.environment.phase == "vote":
            return f"I vote for {self.vote_target}"
        if self.environment.phase == "liar_guess":
            return "Pizza"
        return "A round and savory favorite."


class FakeGradio:
    @staticmethod
    def update(**kwargs):
        return kwargs


def make_game(humans=("Player 1",), liar="Player 5", reveal_topic=True, clue_rounds=2):
    environment = LiarGame(
        player_names=NAMES,
        word_sets={"Food & Drinks": ["Pizza"]},
        clue_rounds=clue_rounds,
        reveal_topic=reveal_topic,
        liar_name=liar,
        topic="Food & Drinks",
        secret_word="Pizza",
    )
    players = [
        FakePlayer(
            name,
            environment,
            human=name in humans,
            vote_target=liar,
        )
        for name in NAMES
    ]
    return HumanLiarGame(Arena(players, environment), humans)


class TestLiarGameGradio(unittest.TestCase):
    @staticmethod
    def view(game, human_name="Player 1"):
        return {
            **game.snapshot(human_name),
            "lobby": False,
            "room_code": "ABCDEF12",
            "seat_code": "1234ABCD",
            "human_count": 1,
            "ai_count": 4,
        }

    @staticmethod
    def config_path():
        return (
            Path(__file__).resolve().parents[2]
            / "examples"
            / "liar_game_qwen3_5_9b.json"
        )

    def test_pilot_config_replaces_one_backend_with_human(self):
        game = create_human_game(str(self.config_path()), 2, 3, False)
        self.assertEqual(game.human_names, {"Player 1", "Player 2"})
        self.assertFalse(game.environment.reveal_topic)
        self.assertEqual(
            sum(player.backend.type_name == "human" for player in game.arena.players),
            2,
        )
        self.assertEqual(
            sum(player.backend.type_name == "openai-chat" for player in game.arena.players),
            3,
        )

    def test_player_counts_are_configurable_with_valid_boundaries(self):
        self.assertEqual(validate_player_counts(1, 2), (1, 2))
        self.assertEqual(validate_player_counts(3, 0), (3, 0))
        self.assertEqual(validate_player_counts(5, 5), (5, 5))
        for humans, ais in ((0, 3), (1, 1), (9, 2), (1.5, 3), (None, 4)):
            with self.subTest(humans=humans, ais=ais):
                with self.assertRaises(ValueError):
                    validate_player_counts(humans, ais)
        all_human_game = create_human_game(str(self.config_path()), 3, 0, True)
        self.assertEqual(all_human_game.environment.num_players, 3)
        self.assertEqual(len(all_human_game.human_names), 3)

    def test_clue_round_count_is_configurable_and_controls_voting_start(self):
        self.assertEqual(validate_clue_rounds(1), 1)
        self.assertEqual(validate_clue_rounds(10), 10)
        for invalid in (0, 11, 1.5, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_clue_rounds(invalid)
        game = create_human_game(str(self.config_path()), 1, 2, True, clue_rounds=3)
        self.assertEqual(game.environment.clue_rounds, 3)
        room = LiarGameRoom(str(self.config_path()), "ROUNDTEST", 1, 2, True, 3)
        token = room.join()
        self.assertEqual(room.view(token)["rounds"], 3)
        self.assertEqual(room.game.environment.clue_rounds, 3)
        one_round = make_game(clue_rounds=1)
        one_round.submit("Player 1", "One clue")
        list(one_round.advance_ai())
        self.assertEqual(one_round.environment.phase, "vote")

    def test_room_waits_for_humans_and_rejoin_invalidates_old_token(self):
        room = LiarGameRoom(str(self.config_path()), "ABCDEF12", 2, 1, False)
        first = room.join(nickname="Alice")
        lobby = room.view(first)
        self.assertTrue(lobby["lobby"])
        self.assertEqual(lobby["joined_humans"], 1)
        self.assertEqual(lobby["seat"], "Seat 1")
        self.assertEqual(lobby["nickname"], "Alice")
        with self.assertRaisesRegex(ValueError, "Waiting"):
            room.submit(first, "A clue")

        second = room.join(nickname="Bob")
        self.assertFalse(room.view(second)["lobby"])
        self.assertEqual(room.view(second)["seat"], "Seat 2")
        self.assertNotEqual(room.view(first)["human_name"], room.view(second)["human_name"])
        self.assertNotEqual(room.view(first)["seat_code"], room.view(second)["seat_code"])
        word = room.game.environment.secret_word
        for token in (first, second):
            view = room.view(token)
            is_liar = view["human_name"] == room.game.environment.liar_name
            self.assertEqual(word in view["role_message"], not is_liar)

        resumed = room.join(lobby["seat_code"])
        self.assertEqual(room.view(resumed)["seat"], "Seat 1")
        self.assertEqual(room.view(resumed)["nickname"], "Alice")
        with self.assertRaises(ValueError):
            room.view(first)

    def test_random_human_slots_rematch_and_identity_reveal(self):
        room = LiarGameRoom(str(self.config_path()), "REMATCH", 2, 1, True, 1)
        with patch("chatarena.ui.liar_game_gradio.secrets.SystemRandom") as random_type:
            random_type.return_value.sample.side_effect = [
                ["Player 3", "Player 1"], ["Player 2", "Player 3"],
            ]
            alice = room.join(nickname="Alice")
            bob = room.join(nickname="Bob")
            self.assertEqual(room.view(alice)["human_name"], "Player 3")
            self.assertEqual(room.view(bob)["human_name"], "Player 1")
            self.assertEqual(room.game.human_names, {"Player 1", "Player 3"})
            self.assertNotIn("participants", room.view(alice))
            with self.assertRaisesRegex(ValueError, "Finish"):
                room.ready_for_restart(alice)
            self.assertFalse(room.step_ai_once())
            environment = room.game.environment
            environment._finish("liar", "Test game finished.")
            room.game.timestep = TimeStep(
                observation=environment.get_observation(),
                reward=environment.get_rewards(), terminal=True,
            )
            revealed = room.view(alice)["participants"]
            first_rewards = environment.get_rewards()
            first_alice_score = first_rewards["Player 3"]
            first_bob_score = first_rewards["Player 1"]
            self.assertEqual(room.view(alice)["my_score"], first_alice_score)
            self.assertEqual(revealed["Player 3"], {"type": "Human", "nickname": "Alice"})
            self.assertEqual(revealed["Player 1"], {"type": "Human", "nickname": "Bob"})
            self.assertEqual(revealed["Player 2"], {"type": "AI", "nickname": ""})
            revealed_panel = render_snapshot(room.view(alice), FakeGradio)[3]
            self.assertIn("Players revealed", revealed_panel)
            self.assertIn("Alice", revealed_panel)
            self.assertIn("Bob", revealed_panel)
            room.ready_for_restart(alice)
            self.assertEqual(room.view(alice)["game_number"], 1)
            self.assertTrue(room.view(alice)["i_am_ready"])
            with self.assertRaisesRegex(ValueError, "Wait for your turn"):
                room.submit(alice, "An extra move")
            room.ready_for_restart(bob)
            self.assertEqual(room.view(alice)["game_number"], 2)
            self.assertEqual(room.view(alice)["human_name"], "Player 2")
            self.assertEqual(room.view(bob)["human_name"], "Player 3")
            self.assertEqual(room.view(alice)["nickname"], "Alice")
            self.assertFalse(room.view(alice)["terminal"])
            self.assertNotIn("participants", room.view(alice))
            self.assertEqual(room.view(alice)["my_score"], first_alice_score)
            self.assertEqual(room.view(bob)["my_score"], first_bob_score)
            self.assertFalse(room.game.next_is_human)  # Player 1 is AI this game.
            with patch.object(
                room.game.arena, "step",
                side_effect=lambda: room.game.environment.step("Player 1", "AI clue"),
            ):
                self.assertTrue(room.step_ai_once())
            self.assertEqual(room.game.environment.get_next_player(), "Player 2")
            self.assertTrue(room.game.next_is_human)
            second_environment = room.game.environment
            second_environment._finish("non_liars", "Test second game finished.")
            room.game.timestep = TimeStep(
                observation=second_environment.get_observation(),
                reward=second_environment.get_rewards(), terminal=True,
            )
            second_rewards = second_environment.get_rewards()
            self.assertEqual(
                room.view(alice)["my_score"],
                first_alice_score + second_rewards["Player 2"],
            )
            self.assertEqual(
                room.view(bob)["my_score"],
                first_bob_score + second_rewards["Player 3"],
            )
            self.assertIn("Room cumulative scores", render_snapshot(
                room.view(alice), FakeGradio,
            )[3])

    def test_nicknames_are_optional_unique_and_html_escaped(self):
        self.assertEqual(validate_nickname("  Alice  "), "Alice")
        with self.assertRaises(ValueError):
            validate_nickname("x" * 33)
        room = LiarGameRoom(str(self.config_path()), "NICKS", 2, 1, True)
        first = room.join(nickname="<Alice>")
        with self.assertRaisesRegex(ValueError, "already in use"):
            room.join(nickname="<alice>")
        second = room.join()
        self.assertEqual(room.view(second)["nickname"], "")
        card = render_snapshot(room.view(first), FakeGradio)[1]
        self.assertIn("&lt;Alice&gt;", card)
        self.assertNotIn("<Alice>", card)

    def test_registry_creates_independent_rooms(self):
        registry = RoomRegistry(str(self.config_path()))
        first_room, first_token = registry.create(1, 4, True)
        second_room, second_token = registry.create(2, 1, False)
        self.assertNotEqual(first_room.room_code, second_room.room_code)
        self.assertFalse(first_room.view(first_token)["lobby"])
        self.assertTrue(second_room.view(second_token)["lobby"])
        with self.assertRaisesRegex(ValueError, "not found"):
            registry.get("WRONG")

    def test_private_role_card_and_public_chat_are_separate(self):
        game = make_game(reveal_topic=False)
        snapshot = game.snapshot("Player 1")
        self.assertIn("Pizza", snapshot["role_message"])
        public = " ".join(content for _, content in snapshot["public_messages"])
        self.assertNotIn("Pizza", public)
        self.assertNotIn("Food & Drinks", public)

        liar_game = make_game(humans=("Player 5",), liar="Player 5")
        liar_snapshot = liar_game.snapshot("Player 5")
        self.assertIn("You are the Liar", liar_snapshot["role_message"])
        self.assertNotIn("Pizza", liar_snapshot["role_message"])

    def test_public_game_info_keeps_topic_and_rules_outside_chat(self):
        game = make_game(reveal_topic=True)
        shown = render_snapshot(self.view(game), FakeGradio)
        self.assertIn("Topic: <b>Food &amp; Drinks</b>", shown[7])
        self.assertIn("A wrong accusation gives the Liar the win", shown[7])
        self.assertNotIn("Topic:", shown[2])
        self.assertNotIn("Secret word: Pizza", shown[7])
        hidden = render_snapshot(self.view(make_game(reveal_topic=False)), FakeGradio)
        self.assertIn("Hidden until the game ends", hidden[7])
        self.assertNotIn("Food &amp; Drinks", hidden[7])

    def test_human_clues_vote_and_ai_turns_finish_the_game(self):
        game = make_game()
        self.assertTrue(game.is_human_turn("Player 1"))
        game.submit("Player 1", "Shared bicycle lanes.")
        self.assertEqual(len(list(game.advance_ai())), 4)
        self.assertTrue(game.is_human_turn("Player 1"))
        game.submit("Player 1", "A second indirect clue.")
        self.assertEqual(len(list(game.advance_ai())), 4)
        self.assertEqual(game.environment.phase, "vote")
        self.assertTrue(game.is_human_turn("Player 1"))
        game.submit("Player 1", vote="Player 5")
        self.assertEqual(game.pending_votes["Player 1"], "I vote for Player 5")
        self.assertEqual(game.environment.votes, {})
        list(game.advance_ai())
        self.assertEqual(game.environment.votes["Player 1"], "Player 5")
        self.assertTrue(game.timestep.terminal)
        self.assertEqual(game.environment.winner, "liar")
        self.assertEqual(game.environment.accused_player, "Player 5")
        self.assertEqual(len(game.environment.votes), 5)

    def test_two_humans_take_distinct_turns_before_ai(self):
        game = make_game(humans=("Player 1", "Player 2"))
        self.assertTrue(game.is_human_turn("Player 1"))
        game.submit("Player 1", "First human clue.")
        self.assertTrue(game.is_human_turn("Player 2"))
        self.assertEqual(list(game.advance_ai()), [])
        with self.assertRaisesRegex(ValueError, "Wait for your turn"):
            game.submit("Player 1", "Out-of-turn clue")
        game.submit("Player 2", "Second human clue.")
        self.assertEqual(len(list(game.advance_ai())), 3)
        self.assertTrue(game.is_human_turn("Player 1"))

    def test_humans_can_vote_in_any_order_without_leaking_ballots(self):
        game = make_game(humans=tuple(NAMES), liar="Player 5", clue_rounds=1)
        for player in NAMES:
            game.submit(player, f"Clue from {player}")
        self.assertEqual(game.environment.phase, "vote")
        self.assertTrue(all(game.is_human_turn(player) for player in NAMES))
        game.submit("Player 5", vote="Player 2")
        self.assertFalse(game.is_human_turn("Player 5"))
        self.assertTrue(game.is_human_turn("Player 1"))
        waiting = game.snapshot("Player 1")
        self.assertEqual(waiting["votes_cast"], 1)
        self.assertEqual(waiting["votes"], {})
        self.assertFalse(waiting["terminal"])
        self.assertNotIn("I vote for Player 2", " ".join(
            content for _, content in waiting["conversation_messages"]
        ))
        waiting_ui = render_snapshot(self.view(game), FakeGradio)
        self.assertIn("1/5 ballots submitted", waiting_ui[3])
        self.assertNotIn("Player 5</td>", waiting_ui[3])
        for player in NAMES[:4]:
            game.submit(player, vote="Player 2" if player != "Player 2" else "Player 1")
        self.assertTrue(game.timestep.terminal)
        self.assertIn("Final votes", render_snapshot(self.view(game), FakeGradio)[3])
        self.assertNotIn("I vote for Player 2", render_snapshot(self.view(game), FakeGradio)[2])

    def test_human_liar_gets_final_guess_turn(self):
        game = make_game(humans=("Player 1",), liar="Player 1")
        game.submit("Player 1", "A plausible clue.")
        list(game.advance_ai())
        game.submit("Player 1", "Another plausible clue.")
        list(game.advance_ai())
        game.submit("Player 1", vote="Player 2")
        list(game.advance_ai())
        self.assertEqual(game.environment.phase, "liar_guess")
        self.assertTrue(game.is_human_turn("Player 1"))
        game.submit("Player 1", "Wrong word")
        self.assertTrue(game.timestep.terminal)
        self.assertEqual(game.environment.winner, "non_liars")

    def test_rejects_empty_clue_and_self_vote(self):
        game = make_game()
        with self.assertRaisesRegex(ValueError, "Enter a clue"):
            game.submit("Player 1", "   ")
        with self.assertRaisesRegex(ValueError, "Wait for your turn"):
            game.submit("Player 2", "Impersonated clue")
        game.submit("Player 1", "A clue")
        list(game.advance_ai())
        game.submit("Player 1", "Another clue")
        list(game.advance_ai())
        with self.assertRaisesRegex(ValueError, "other player"):
            game.submit("Player 1", vote="Player 1")
        self.assertEqual(game.environment.phase, "vote")

    def test_html_render_escapes_public_player_text(self):
        game = make_game()
        game.submit("Player 1", "<script>alert('x')</script>")
        snapshot = {
            **game.snapshot("Player 1"),
            "lobby": False,
            "room_code": "ABCDEF12",
            "seat_code": "1234ABCD",
            "human_count": 1,
            "ai_count": 4,
        }
        rendered = render_snapshot(snapshot, FakeGradio)
        self.assertIn("&lt;script&gt;", rendered[2])
        self.assertNotIn("<script>", rendered[2])
        self.assertNotIn("The secret word is Pizza", rendered[2])
        self.assertNotIn("value", rendered[4])
        self.assertNotIn("value", rendered[5])
        cleared = render_snapshot(snapshot, FakeGradio, clear_input=True)
        self.assertEqual(cleared[4]["value"], "")
        self.assertIsNone(cleared[5]["value"])

    def test_chat_shows_recent_player_messages_first_without_moderator(self):
        game = make_game()
        snapshot = self.view(game)
        snapshot["public_messages"] = [
            ("Moderator", "Instruction for another player"),
            *[("Player 2", f"clue {number}") for number in range(12)],
            ("Moderator", "The Liar wins. Secret word: Pizza"),
        ]
        snapshot["conversation_messages"] = [
            (speaker, content) for speaker, content in snapshot["public_messages"]
            if speaker != "Moderator"
        ]
        chat = render_snapshot(snapshot, FakeGradio)[2]
        self.assertNotIn("Moderator", chat)
        self.assertNotIn("Instruction for another player", chat)
        self.assertNotIn("Secret word", chat)
        self.assertLess(chat.index("clue 11"), chat.index("clue 10"))
        self.assertIn("Show earlier messages", chat)
        self.assertLess(chat.index("Show earlier messages"), chat.index("clue 0"))

    def test_only_current_humans_instruction_is_shown(self):
        game = make_game()
        own = render_snapshot(self.view(game), FakeGradio)
        self.assertIn("Your instruction:", own[0])
        game.submit("Player 1", "An indirect clue")
        waiting = render_snapshot(self.view(game), FakeGradio)
        self.assertNotIn("Your instruction:", waiting[0])
        self.assertIn("Moderator · to you", waiting[2])
        self.assertNotIn("Player 2, give one", waiting[2])

    def test_own_voting_and_final_guess_instructions_remain_visible(self):
        game = make_game(liar="Player 1")
        game.submit("Player 1", "First clue")
        list(game.advance_ai())
        game.submit("Player 1", "Second clue")
        list(game.advance_ai())
        vote_view = render_snapshot(self.view(game), FakeGradio)
        self.assertIn("Your instruction:", vote_view[0])
        self.assertIn("vote privately for one other", vote_view[0])
        game.submit("Player 1", vote="Player 2")
        list(game.advance_ai())
        guess_view = render_snapshot(self.view(game), FakeGradio)
        self.assertEqual(self.view(game)["votes"], {})
        self.assertIn("All ballots are sealed", guess_view[3])
        self.assertIn("Your instruction:", guess_view[0])
        self.assertIn("final guess", guess_view[0])
        self.assertIn("Moderator · to you", guess_view[2])
        self.assertNotIn("Player 2, give one", guess_view[2])

    def test_result_is_separate_from_chat_and_secret_is_terminal_only(self):
        game = make_game()
        before = self.view(game)
        self.assertIsNone(before["result"])
        game.submit("Player 1", "First clue")
        list(game.advance_ai())
        game.submit("Player 1", "Second clue")
        list(game.advance_ai())
        game.submit("Player 1", vote="Player 5")
        list(game.advance_ai())
        after = self.view(game)
        self.assertTrue(after["terminal"])
        self.assertEqual(after["result"]["secret_word"], "Pizza")
        rendered = render_snapshot(after, FakeGradio)
        self.assertIn("Secret word: Pizza", rendered[3])
        self.assertIn("Actual Liar: Player 5", rendered[3])
        self.assertNotIn("Secret word:", rendered[2])
        self.assertIn("Moderator · to you", rendered[2])
        self.assertNotIn("Secret word: Pizza", rendered[2])


if __name__ == "__main__":
    unittest.main()
