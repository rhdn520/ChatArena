"""Logging integration checks use a mock API, not model validation."""
import importlib.util
import json
from pathlib import Path

import httpx
import openai
import pytest


@pytest.mark.parametrize("game,moderator", [("mafia", True), ("mafia", False), ("liar_game", False)])
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_complete_game_logs_with_mock_api(tmp_path, monkeypatch, game, moderator, enable_thinking):
    from chatarena.backends import openai as backend

    def respond(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "mock-8b", "object": "model", "created": 0, "owned_by": "test"}]})
        body = json.loads(request.content)
        assert body["chat_template_kwargs"]["enable_thinking"] is enable_thinking
        instruction = body["messages"][-1]["content"]
        if enable_thinking:
            assert "stop" not in body
            assert all("<EOS>" not in m["content"] for m in body["messages"])
            control = instruction.startswith(("REQUEST intent:", "REQUEST moderation:"))
            assert body["thinking_token_budget"] == (256 if control else 2048)
            assert {name: body[name] for name in ("temperature", "top_p", "top_k", "min_p",
                "presence_penalty", "repetition_penalty")} == {
                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "presence_penalty": 1.5, "repetition_penalty": 1.0}
        content = "0" if "digit 0" in instruction or "REQUEST intent" in instruction else "I vote for Player 2"
        if "REQUEST game_action" in instruction:
            # This fixture exercises logging, not language quality. Include
            # all names so the engine finds an eligible target in every phase.
            content = "Player 1 Player 2 Player 3 Player 4"
        if "REQUEST moderation" in instruction:
            content = "END_DISCUSSION"
        return httpx.Response(200, json={
            "id": "mock-call", "object": "chat.completion", "created": 0, "model": "mock-8b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content,
                         "reasoning": "Private reasoning example" if enable_thinking else None},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.setattr(backend, "client", openai.OpenAI(
        api_key="mock", base_url="http://mock.test/v1", http_client=httpx.Client(
            transport=httpx.MockTransport(respond))), raising=False)
    monkeypatch.setattr(backend, "is_openai_available", True)
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_local_game_logs.py"
    spec = importlib.util.spec_from_file_location("local_game_logs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / game
    argv = [str(script), "--game", game, "--base-url",
                                    "http://mock.test/v1", "--model", "mock-8b",
                                    "--players", "4" if game == "mafia" else "5",
                                    "--seeds", "1", "--output", str(output)]
    if game == "mafia" and not moderator:
        argv.append("--no-moderator")
    if enable_thinking:
        argv.extend(["--enable-thinking", "--max-tokens", "4096", "--request-timeout", "300",
                     "--thinking-token-budget", "2048", "--control-thinking-token-budget", "256",
                     "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0",
                     "--presence-penalty", "1.5", "--repetition-penalty", "1.0"])
    monkeypatch.setattr("sys.argv", argv)
    module.main()
    episode = output / "seed_1"
    summary = json.loads((episode / "summary.json").read_text())
    assert summary["terminal"] and summary["error"] is None
    messages = json.loads((episode / "messages.json").read_text())
    assert any(m["visible_to"] != "all" for m in messages)
    assert "비밀 →" in (episode / "conversation.txt").read_text()
    assert "비밀 →" not in (episode / "public_conversation.txt").read_text()
    calls = [json.loads(line) for line in (episode / "api_calls.jsonl").read_text().splitlines()]
    assert len(calls) == summary["api_calls"]
    assert all(c["response"]["choices"][0]["finish_reason"] == "stop" for c in calls)
    assert "[원문 API 응답]" in (episode / "model_io.txt").read_text()
    assert summary["moderator_enabled"] == moderator
    assert summary["enable_thinking"] is enable_thinking
    reasoning = (episode / "reasoning.txt").read_text()
    assert bool(reasoning) is enable_thinking
    assert summary["reasoning_calls"] == (len(calls) if enable_thinking else 0)
    assert "Private reasoning example" not in (episode / "conversation.txt").read_text()
    if enable_thinking:
        assert summary["max_tokens"] == 4096 and summary["request_timeout"] == 300
        assert summary["thinking_token_budget"] == 2048
        assert summary["control_thinking_token_budget"] == 256
        assert summary["sampling_parameters"]["presence_penalty"] == 1.5
    decisions = json.loads((episode / "decisions.json").read_text())
    moderation = [r for r in decisions if r["request"]["kind"] == "moderation"]
    assert bool(moderation) == moderator
    for record in moderation:
        assert all(m["visible_to"] == "all" for m in record["request"]["observation"])
    assert summary["decision_schema_version"] == 2
    assert all("trainable" not in record and "format_eligible" in record
               and record["expert_approved"] is False for record in decisions)
    for call in calls:
        system = call["request"]["messages"][0]["content"]
        if game == "mafia":
            if "Your name is Moderator." in system:
                assert "비밀 역할:" not in system
                assert "조사 결과:" not in system
            else:
                assert "[게임 엔진이 제공한 본인 전용 상태" in system
                assert "[게임 엔진이 제공한 본인 전용 상태" in call["request"]["messages"][-1]["content"]
        else:
            assert "[Private engine state" in system
        if "Private engine state" in system:
            # The Liar must not acquire a non-liar's private word via system state.
            if "Your role is Liar." in system:
                assert "Your private secret word:" not in system
    assert ("END_DISCUSSION" in (episode / "moderator_decisions.txt").read_text()) == moderator
