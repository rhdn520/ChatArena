# 마피아 환경과 RL 구현 안내

현재 구현은 낮 토론을 **발언 의향 기반**으로 실행한다. 이전의 순서제 토론 및 발언 횟수에 대한 보상 설명은 적용하지 않는다.

전체 설계·설정·실행·정보 공개 경계·테스트는 [마피아 토론 파이프라인](devdoc/mafia_discussion_pipeline.md)을 참고한다.

## 게임과 학습 흐름

1. 역할을 배정하고 밤의 마피아·의사·경찰 행동을 처리한다.
2. 낮에는 생존 AI별로 0~3 의향 점수를 수집하고 가중 추첨으로 발언자를 정한다. 0점과 PASS는 유효한 침묵이다.
3. 선택적 사회자가 토론 진행을 판단한다. 침묵이나 제한 도달 시 투표로 전환한다.
4. 기존 투표·승패 규칙에 따라 게임을 진행한다.
5. 완료된 에피소드의 팀 승패 보상을 이용해 의향 출력과 실제 행동을 함께 학습한다.

`MafiaRolloutManager`와 `Arena`는 같은 토론 제어기를 사용한다. RL은 공유 모델의 안전을 위해 callback을 직렬 호출하고, 웹은 API 요청을 병렬 실행한다.

## 보상과 손실

승리 팀 +1, 패배 팀 -1, 무승부 0이다. 사망한 팀원도 같은 보상을 받는다. 과거의 발언 보너스·형식 보너스는 제거했다.

그룹 내 보상으로 advantage를 구한 후, 각 결정의 **응답 토큰 평균 log-probability**, 플레이어 결정 평균, 그룹 평균 순서로 policy gradient 손실을 계산한다. 정상 의향 출력은 발언자로 뽑히지 않아도 학습한다. 오류·재시도·오래된 응답·사회자 출력은 학습하지 않는다.

미완료 에피소드는 `winner="truncated"`이며 학습과 완료 게임 승률에서 제외한다. 최대 일수로 끝난 실제 무승부와 구별한다. 이 트레이너는 그룹 정규화 policy gradient이며 PPO clipping/reference KL은 구현하지 않는다.

## 실행

```bash
conda run -n chatarena_37 python training/train_mafia_rl.py \
  --dry_run --num_episodes 4 --group_size 2 \
  --max_discussion_messages 3 --seed 17
```

실제 GPU 학습은 PyTorch, Transformers, PEFT, Accelerate 설치 후 `--dry_run`을 제거하고 모델·장치·LoRA 옵션을 지정한다. 플레이어 수는 `--num_players` 또는 `--dynamic_players --min_players 4 --max_players 6`으로 설정한다. `--group_size`는 업데이트에 수집하는 에피소드 수다.

출력 디렉터리의 `episodes.jsonl`에서 의향 점수 분포와 발언·패스·실패 비율, 종료 사유를 확인할 수 있다. 모델의 실제 자연스러움이나 전략 향상은 mock 테스트만으로 보장하지 않는다.
