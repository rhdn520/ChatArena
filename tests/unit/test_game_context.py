"""Regressions for identity refresh, private facts, and phase enforcement."""
from unittest.mock import Mock, patch

import pytest

from chatarena.agent import Player
from chatarena.arena import Arena
from chatarena.backends import openai as backend
from chatarena.environments.liar_game import LiarGame
from chatarena.environments.mafia import Mafia
from chatarena.mafia_discussion import MafiaDiscussionController


NAMES = list("ABCDEFG")
ROLES = dict(zip(NAMES, ["mafia", "mafia", "doctor", "police", "villager", "villager", "villager"]))


def daytime():
    env = Mafia(NAMES, role_mapping=ROLES)
    env.end_discussion("test")
    env.step("A", "E")
    env.step("B", "E")
    env.step("C", "E")
    env.step("D", "A")
    return env


def liar_env(reveal_topic=True):
    return LiarGame(["A", "B", "C"], word_sets={"Places": ["Amusement Park"]},
                    liar_name="C", topic="Places", secret_word="Amusement Park",
                    clue_rounds=1, reveal_topic=reveal_topic)


def test_police_result_is_refreshed_only_for_its_recipient():
    env = daytime()
    result = "조사 결과: **A** — **마피아**."
    assert result in env.get_agent_context("D")
    for name in "ABCEFG":
        assert result not in env.get_agent_context(name)
    env.discussion_speak("F", "조사 결과: **B** — **마피아**. 조사를 했습니다.")
    assert "조사 결과: **B**" not in env.get_agent_context("D")
    env.reset()
    assert result not in env.get_agent_context("D")


def test_mafia_context_separates_private_and_public_audiences():
    env = Mafia(NAMES, role_mapping=ROLES)
    private = env.get_agent_context("A")
    assert "비밀 역할: 마피아" in private and "동료 마피아: B" in private
    assert "생존 동료 마피아만" in private
    assert "동료 마피아:" not in env.get_agent_context("G")
    env = daytime()
    public = env.get_agent_context("A")
    assert "공개, 모든 플레이어" in public
    assert "비공개 작전을 공개 토론에 옮기지" in public
    env.end_discussion("test")
    assert "선택 가능한 대상: A, B, C, D, E, F, G" in env.get_agent_context("A")


def test_moderator_receives_only_current_public_discussion_context():
    env = daytime()
    env.discussion_speak("E", "A의 근거를 듣고 싶습니다.")
    env.discussion_speak("F", "같은 의심만 반복되고 있습니다.")
    ctrl = MafiaDiscussionController(env, lambda req: "0", moderator_query=lambda req: "CONTINUE")
    try:
        req = ctrl._request("Moderator", "moderation")
        assert all(m.visible_to == "all" for m in req.observation)
        assert "E: A의 근거" in req.instruction
        assert "E: A의 근거" not in req.game_context
        assert "조사 결과:" not in req.game_context
        assert "당신의 비밀 역할" not in req.game_context
        assert "동료 마피아:" not in req.game_context
        assert "표현만 바꾸거나 동의하는 것은 새 정보가 아닙니다" in req.instruction
        assert "기본 응답은 CONTINUE" not in req.instruction
    finally:
        ctrl.close()


def test_actual_api_rendering_puts_own_state_in_initial_system_and_last_request():
    env = daytime()
    ctrl = MafiaDiscussionController(env, lambda req: "0")
    try:
        req = ctrl._request("D", "speech")
        with patch.object(backend, "is_openai_available", True):
            model = backend.OpenAIChat(model="mock", system_messages_at_start=True)
        player = Player("D", "Original description", model)
        with patch.object(model, "_get_response", return_value="A를 의심합니다.") as generate:
            from chatarena.message import Message
            player.act(list(req.observation), Message("System", req.instruction, -1),
                       game_context=req.game_context)
        messages = generate.call_args.args[0]
        assert [m["role"] for m in messages].count("system") == 1
        for text in [messages[0]["content"], messages[-1]["content"]]:
            assert "당신은 D입니다. 비밀 역할: 경찰" in text
            assert "조사 결과: **A** — **마피아**." in text
            assert "공개, 모든 플레이어" in text
        assert player.role_desc == "Original description"
    finally:
        ctrl.close()


def test_liar_context_keeps_word_role_and_hidden_topic_private():
    for reveal in (True, False):
        env = liar_env(reveal)
        assert "Amusement Park" in env.get_agent_context("A")
        assert "Amusement Park" not in env.get_agent_context("C")
        assert "Your role is Liar" in env.get_agent_context("C")
        assert ("Places" in env.get_agent_context("C")) == reveal
        assert "non-liar" not in env.get_agent_context("C")


def test_forbidden_clue_is_rejected_without_committing_or_revealing_word():
    env = liar_env()
    before = list(env.get_observation())
    assert not env.check_action("A large AMUSEMENT-PARK.", "A")
    with pytest.raises(ValueError) as error:
        env.step("A", "A large AMUSEMENT-PARK.")
    assert "Amusement Park" not in str(error.value)
    assert env.get_observation() == before
    assert env.get_next_player() == "A"
    env.step("A", "A place to spend a day.")
    env.step("B", "A venue with tickets.")
    assert not env.check_action("I think amusement park fits.", "C")
    assert env.check_action("Somewhere people visit.", "C")


def test_arena_retries_rejected_clue_with_private_state_and_no_public_commit():
    env = liar_env()
    a = Mock(name="A")
    a.name = "A"
    a.act.side_effect = ["Amusement Park", "Somewhere with an entrance."]
    arena = Arena([a], env)
    arena.step()
    assert a.act.call_count == 2
    assert "previous response was rejected" in a.act.call_args.args[1].content
    public = [m.content for m in env.get_observation() if m.visible_to == "all"]
    assert "Amusement Park" not in public
    assert "Somewhere with an entrance." in public
    assert len(arena.action_rejections) == 1
    assert arena.action_rejections[0]["response"] == "Amusement Park"
    arena.reset()
    assert arena.action_rejections == []


def test_vote_parser_does_not_select_a_name_from_a_negation_or_multiple_targets():
    env = liar_env()
    assert env._parse_vote("I vote for B", "A") == "B"
    assert env._parse_vote("I do not vote for B; I vote for C", "A") is None
    assert env._parse_vote("I vote for B or C", "A") is None
    assert env._parse_vote("I vote for A", "A") is None


def test_api_history_marks_audience_without_disclosing_private_recipient_groups():
    from chatarena.message import Message
    history = [Message("A", "night plan", 1, visible_to=["A", "B"]),
               Message("A", "public claim", 2),
               Message("System", "private instruction", 3, visible_to=["A"])]
    rendered = backend.OpenAIChat.format_messages(
        "A", "rules", history, system_messages_at_start=True)
    assert "[Private: visible to you; not public]" in rendered[1]["content"]
    assert "[Public: all players]" in rendered[2]["content"]
    assert "[Private: visible to you; not public]" in rendered[3]["content"]
    assert "B" not in rendered[1]["content"]
    assert rendered[1]["role"] == rendered[2]["role"] == "assistant"
    env = liar_env()
    messages = backend.OpenAIChat.format_messages("A", "rules", env.get_observation("A"))
    secret = next(m for m in messages if "The secret word is" in m["content"])
    assert "[Private: visible to you; not public]" in secret["content"]
    assert "B" not in secret["content"] and "C" not in secret["content"]
    liar_messages = backend.OpenAIChat.format_messages("C", "rules", env.get_observation("C"))
    assert all("Amusement Park" not in m["content"] for m in liar_messages)


def test_day_speech_instruction_distinguishes_role_specific_disclosure():
    env = daytime()
    ctrl = MafiaDiscussionController(env, lambda req: "0")
    try:
        mafia = ctrl._request("A", "speech")
        police = ctrl._request("D", "speech")
        doctor = ctrl._request("C", "speech")
        assert "지난밤의 결과를 공개하거나" not in mafia.instruction
        assert "시민으로 위장" in mafia.instruction
        assert "밤 공격 의논·투표를 확인해 주지" in mafia.instruction
        assert "실제 조사 결과를 공개할지는" in police.instruction
        assert "조사 결과: **A** — **마피아**." not in mafia.instruction
        assert "보호에 성공했다고 알 수는 없습니다" in doctor.instruction
    finally:
        ctrl.close()


def test_liar_receives_role_appropriate_clue_task_and_complete_win_condition():
    env = liar_env()
    initial = env.get_observation("C")
    assert any("one final word guess" in m.content for m in initial)
    assert "Liar fails the final word guess" in env.get_agent_context("A")
    env.step("A", "A place with tickets.")
    env.step("B", "It can be crowded.")
    instruction = env.get_observation("C")[-1].content
    assert "You were not given the word" in instruction
    assert "show you know" not in instruction
    assert "Amusement Park" not in instruction
    env.step("C", "People visit with friends.")
    assert "ALL rounds" in env.get_agent_context("A")
    assert "divert the accusation" in env.get_agent_context("C")
    assert "Amusement Park" not in env.get_agent_context("C")
