# 최소 변경·원본 보존·작업 공개 — 2026-10-17

전역 `/home/jaehoonjeong/.codex/AGENTS.md`의 최소 변경·작업 공개 절차를 적용한다.
각 명령·수정 전에 대상·목적·범위를 일반 commentary로 설명하고 결과·실패·미확인 사항을 보고한다.
프롬프트 개선은 기존 전달 위치의 최소 수정부터 검토한다. 상태·규칙의 중복 주입, 전략 코칭, 사례별 예외를 덧붙이지 않는다.
범위 밖의 Player/Arena/백엔드/RL 공통 API, 발언자 선택, 종료 절차, 게임 규칙 변경은 구체적 변경안과 해당 범위의 명시 동의가 먼저다.
라이어 비교 기준은 origin/liargame의 ebc2e9d, 마피아 비교 기준은 origin/realistic_mafia의 03b774c다. 이 기준은 자동 원복 명령이 아니며 사용자·동료의 후속 편집을 덮어쓰지 않는다.
CURRENT_WORK.md의 과거 개선 계획을 새 작업 허가로 해석하지 않는다. 현재 사용자 요청과 최신 명시적 정정이 우선한다.
새 검사기·훅·게이트를 추가하지 않는다. 지침 반영과 실제 코드 복구·실행 성공을 구분해 보고한다.

# 테스트·검사기 정책 — 사용자 명시 지시, 2026-10-15

- 훅은 Slurm 자원·제출 규격 확인용으로만 사용한다.
- 그 외 테스트·검사기·preflight·smoke·모델 로딩 점검·응답 형식 검사·검증 게이트는 사용자의 명시 요청 없이 만들거나 추가·확대·실행하지 않는다.
- 실행·재실행·수정·확인 요청을 테스트나 검사기 제작·실행 허가로 확대 해석하지 않는다. 확인 요청에는 기존 결과와 로그를 읽어 답한다.
- 검사 필요성을 자의적으로 판단하여 작업을 지연하거나 실행을 차단하지 않는다. 삭제한 검사를 되살리거나 이름만 바꿔 재도입하지 않는다.
- 테스트 개수나 검사 통과를 실제 실행 성공 대신 보고하지 않는다. 관측한 실행 상태와 한계를 그대로 보고한다.
- 이 정책은 사용자 관리 지침에 남아 있던 자동 사전검사·스모크·실행 게이트 강제 규정을 대체한다. Slurm 규격 확인은 유지한다.
- 기존 실행환경·경로·모델·자원·로그를 보존한다. 사용자 지시 없는 폴링 금지.

# ChatArena execution

Read CURRENT_WORK.md and preserve the user's latest requested task and corrections.
Use the existing .venv-qwen35, server launcher, cache/output/log paths and live workspace.
Preserve model, thinking setting, game rules and private-information boundaries.
Report full job specifications before submission. Use scripts/submit_qwen35_games.py.
This workload uses one A6000, one node, two CPU cores, 32G RAM, 48 hours and the default partition.
The submit wrapper and hooks check only Slurm resource/directive specifications.
No CPU model-load gate, CUDA probe, GPU smoke, separate API smoke or response-format gate.
Do not automatically resubmit a failed job without a new user instruction.
Report scheduler state and actual outputs accurately; submission is not completed inference.
Keep CURRENT_WORK.md current. No polling, execution snapshots, source copies or checksums.
