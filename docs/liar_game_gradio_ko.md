# Liar Game: 사용법 안내

이 문서는 Qwen3.5-9B AI와 사람이 함께 플레이하는 Gradio 버전의 실행 방법이다. 
**잡 제출자**는 클러스터에서 모델과 웹 UI를 띄우고 방을 만든다. 
**참가자**는 각자 자신의 컴퓨터에서 SSH 터널을 열어 같은 UI에 접속한다. 방 코드만으로는 네트워크 접속이 되지 않는다.

아래 명령의 `<JOB_ID>`, `<USER>`, `<LOGIN_HOST>`, `<COMPUTE_NODE>`, `<GRADIO_PORT>`는 실제 값으로 **바꿔서** 실행한다.

## 1. 잡 제출자: 최초 준비 -- 절대 경로 수정 필요!!

현재 스크립트는 `/home/solbeecho/data/ChatArena`, `/home/solbeecho/data/conda_envs/`, `/home/solbeecho/data/datasets/huggingface_cache` 등 **이 클러스터의 절대경로**를 사용한다. 다른 계정·서버에서 실행하려면 `scripts/*.sh`의 `#SBATCH --output/--error`, `PROJECT_DIR`, 콘다 경로, 모델 캐시 경로와 `source` 경로를 먼저 자신의 `/home/<사용자>/data/` 아래 경로로 바꿔야 한다. 특히 `scripts/download_qwen3_5_9b.sh`는 `mhv_env`에 `hf` 명령이 설치돼 있다고 가정한다. 저장소만 복제하면 콘다 환경이나 모델 가중치가 자동으로 따라오지는 않는다.

현재 계정·클러스터에서 처음 준비할 때는 다음 잡을 각각 제출한다. **각 잡의 완료 상태와 오류 로그를 확인한 뒤** 다음 단계로 진행한다. Python 파일을 로그인 노드에서 직접 실행하지 않는다.

```bash
cd ChatArena
mkdir -p logs
sbatch scripts/create_qwen3_5_runtime.sh
sbatch scripts/setup_liar_game_gradio.sh
sbatch scripts/download_qwen3_5_9b.sh
```

각 `sbatch`가 출력한 잡 번호로 확인한다. `Submitted batch job ...`은 완료를 뜻하지 않는다.

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode
```

정상 완료는 `COMPLETED`와 `0:0`이다. 실패 시 `logs/<JOB_ID>-*.err`를 확인한다. 모델은 `/home/solbeecho/data/datasets/huggingface_cache`에 다운로드된다. 준비가 이미 끝났다면 위 세 잡을 다시 제출할 필요는 없다.

## 2. 잡 제출자: 게임 서버와 방 만들기

```bash
cd /home/solbeecho/data/ChatArena
sbatch scripts/run_liar_game_gradio.sh
```

출력된 잡 번호를 `<JOB_ID>`라 하면, 다음 로그에서 계산 노드와 Gradio 포트를 확인한다.

```bash
squeue -j <JOB_ID>
tail -f logs/<JOB_ID>-liar-game-gradio.out
```

로그에 `Compute node: n02`, `Gradio port: 21794`처럼 표시된다. **이 값은 예시이며 매 잡마다 달라진다.** `Gradio listening on compute-node localhost:...`가 나와야 웹 UI가 시작된 것이다. 오류가 나면 `logs/<JOB_ID>-liar-game-gradio.err`와 `logs/<JOB_ID>-liar-game-vllm.log`를 확인한다.

잡 제출자도 자신의 **로컬 컴퓨터 터미널**에서 아래 3절의 SSH 터널을 열고 브라우저에 접속한다. `Create room`에서 사람 수(잡 제출자 포함), AI 수, 단서 라운드 수, 주제 공개 여부와 자신의 닉네임을 정한다. 생성된 **방 코드**를 참가자에게 전달한다. 개인 **재입장 코드**는 공유하지 않는다.

## 3. 참가자: 자기 컴퓨터에서 접속하기

잡 제출자에게 **계산 노드**, **Gradio 포트**, **방 코드**를 받는다. 클러스터 접속 권한이 있는 참가자는 자신의 Mac/PC **로컬 터미널**에서 다음처럼 터널을 연다. `<LOGIN_HOST>`는 클러스터 로그인 서버 주소이고 `<USER>`는 **참가자 자신의 클러스터 계정**이다.

```bash
ssh -J <USER>@<LOGIN_HOST> -N -L 7860:127.0.0.1:<GRADIO_PORT> <USER>@<COMPUTE_NODE>
```

예를 들어 로그의 계산 노드가 `n02`, 포트가 `21794`라면 마지막 부분은 `<USER>@n02`, `7860:127.0.0.1:21794`가 된다. 터널 명령은 게임을 하는 동안 종료하지 않는다. 각 참가자는 자기 컴퓨터에서 같은 명령을 실행할 수 있으며 로컬 포트 `7860`이 이미 쓰이고 있다면 `7861` 등 다른 포트를 골라도 된다.

그 컴퓨터의 브라우저에서 `http://127.0.0.1:7860`을 연다. `Join room`에 방 코드를 넣고 원하면 닉네임을 설정한다. **VS Code 원격 터미널에서 SSH 터널을 열면 `127.0.0.1`은 원격 컴퓨터를 가리킨다.** 로컬 Safari/Chrome에서 접속하려면 터널을 로컬 컴퓨터에서 열어야 한다.

참가자에게 클러스터 계정·계산 노드 SSH 권한이 없다면 이 방식으로는 접속할 수 없다. 현재 서버는 `127.0.0.1`에만 바인딩되고 Gradio 공개 공유 링크도 만들지 않는다. 클러스터 정책에 맞는 별도 프록시가 필요하다. SSH 접속 정보나 개인 재입장 코드를 다른 사람과 공유하지 않는다.

## 4. 방 운영과 재시작

- 사람 수를 `2`로 만들면 **방장 포함 2명**만 입장한다. 세 번째 사람은 좌석이 꽉 찼다는 오류를 받는다. 사람 수는 방 생성 뒤에 늘릴 수 없으므로 3명이 플레이하려면 사람 수 `3`으로 새 방을 만든다.
- 누군가 연결이 끊기면 같은 방 코드와 **자기 개인 재입장 코드**로 기존 좌석에 다시 들어간다. 이 코드를 아는 다른 사람이 들어오면 기존 세션이 무효화되므로 비밀로 유지한다.
- 각 판이 끝나면 사람 참가자 **전원이** `Ready for next game`을 눌러야 같은 방에서 새 판이 시작된다. 새 판마다 사람과 AI의 `Player 1`, `Player 2` 등 번호를 다시 무작위로 배정한다. 무작위이므로 같은 사람이 연속으로 같은 번호를 받을 수는 있다.
- 누적 점수는 바뀌는 Player 번호가 아니라 참가자의 **방 내 좌석**에 기록된다. 종료 화면에서 그 판의 사람/AI 배치와 닉네임을 볼 수 있다. 투표 선택과 집계는 최종 결과에서만 공개된다.
- 방과 누적 점수는 **서버 메모리에만** 있다. 잡이 종료되거나 서버가 재시작되면 사라지며, 실제 신원 인증이나 영구 점수 파일은 현재 없다. 닉네임만으로 사람을 인증하지 않는다.

잡을 끝낼 때는 잡 제출자가 `scancel <JOB_ID>`를 실행한다. 진행 중인 방과 누적 점수가 사라지므로 참가자에게 먼저 알리는 편이 좋다.