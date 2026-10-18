# 로컬 8B 모델 대화 로그 검토

현재 실험 모델은 사용자 요청에 따라 **Qwen/Qwen3.5-9B**로 변경했다.
Qwen3-8B 캐시는 삭제했다. 모든 플레이어·사회자 API 요청과 서버 기본 설정에
2026-10-15 사용자 요청으로 새 제출은 `enable_thinking=true`를 적용한다.
기존 268515/268867 결과는 thinking OFF이며 새 설정의 결과가 아니다.
마피아 기본 인원은 7명, 라이어는 5명이다.
전용 실행 환경은 `.venv-qwen35`다.
`scripts/slurm/run_qwen35_games.sh`는 GPU 생성 smoke 후 각 게임 3판을 실행한다.
제출 절차와 검증 범위는 `docs/slurm_policy.md`를 따른다.

로컬 `local-evaluation` 브랜치는 `realistic_mafia`의 최신 마피아 구현에
`liargame`의 환경·단어 목록·기본 설정·환경 테스트를 추가한다.
원격 브랜치는 변경하지 않는다. 실행 대상은 현재 워크스페이스다.
현재 마피아 기준은 `realistic_mafia`의 `03b774c` 업데이트다.

## 실행

실행 중인 OpenAI 호환 모델 서버의 `/v1` 주소와 served model ID가 필요하다.
이 스크립트는 GPU 서버를 시작하거나 Slurm 잡을 제출하지 않는다.
먼저 `/models`와 실제 한 번의 생성을 확인하고 게임을 진행한다.
이 API 확인은 GPU 또는 모델 로딩 preflight를 대체하지 않는다.
마피아는 기본으로 LLM moderator를 켜고 플레이어와 같은 모델을 사용한다.
사회자는 공개 대화만 보고 CONTINUE/END_DISCUSSION으로 낮 토론 종료를
판단한다. 비공개 밤 의논에는 참여하지 않는다. `--moderator-model ID`로
같은 API의 다른 모델을 지정하거나 `--no-moderator`로 비교 실행할 수 있다.

```bash
.venv-qwen35/bin/python scripts/run_local_game_logs.py \
  --game mafia --base-url http://HOST:PORT/v1 --model SERVED_MODEL_ID \
  --seeds 1 2 3 --output outputs/mafia_8b_first

.venv-qwen35/bin/python scripts/run_local_game_logs.py \
  --game liar_game --base-url http://HOST:PORT/v1 --model SERVED_MODEL_ID \
  --seeds 1 2 3 --output outputs/liar_8b_first
```

각 판의 `conversation.txt`를 처음부터 읽는다. 공개 대화와 비밀 메시지를
수신자 표시로 구분한다. `public_conversation.txt`는 공개 대화만 담는다.
`model_io.txt`는 실제 API 입력과 원문 응답·종료 사유·토큰 사용량을 담는다.
`api_calls.jsonl`, `messages.json`, `decisions.json`은 기계 판독용 기록이다.
`reasoning.txt`는 API가 반환한 추론과 최종 응답을 별도로 모은다. 추론은
공개 대화나 다른 플레이어의 관측으로 전달하지 않는다.
API 호출은 번호로 시작 순서를 표시하며 JSONL 저장 순서는 완료 순서다.
마피아 발언 의향과 PASS는 공개 대화에 나타나지 않으므로 `decisions.json`도
참고한다. 실패하거나 step 상한에 걸려도 부분 대화와 오류 요약을 저장한다.
`moderator_decisions.txt`는 공개 대화에 나오지 않는 사회자의 판단을 담는다.
이미 있는 출력 디렉터리는 덮어쓰지 않는다. seed는 게임 무작위성만 제어한다.

## 읽을 때 확인할 것

- 단계 이해: 낮에 밤 행동을 하거나 단서 차례에 투표하지 않는가?
- 관측 이해: 다른 사람의 발언에 답하고, 자신이 받은 역할·조사 결과를 기억하는가?
- 정보 경계: 접근할 수 없는 비밀을 사실처럼 말하는가? 마피아의 전략적 거짓말과 구분한다.
- 출력 형식: 의향 0~3, 투표 대상, PASS, 최종 단어 추측을 지키는가?
- 대화 품질: 반복·혼잣말·사칭 없이 질문과 반박을 이어가는가?

최신 마피아 `_parse_target`은 대상 인식 실패 시 None을 반환하며 임의 대상을
선택하지 않는다. 대상 이름의 존재만으로 엄격한 문장 형식 준수를 보장하지는 않는다.
라이어 `_parse_vote`도 지정 문장 형식보다 느슨하게 이름을 인식한다.
완주나 승률만으로 규칙 준수를 판정하지 말고 원문과 함께 검토한다.
라이어 기본 프롬프트와 단어는 영어, 최신 마피아는 한국어이므로 언어 차이도
감안한다. 먼저 원 구현 기준으로 확인하고 필요하면 한국어 조건을 별도로 비교한다.

몇 판을 읽어 반복 실패를 분류한 뒤 프롬프트·토큰 한도·API 형식 문제를
수정하고 같은 조건에서 재평가한다. 그 뒤에도 역할·단계 이해나 대화 연결
실패가 반복되는지 보고 imitation learning 필요성을 판단한다.
