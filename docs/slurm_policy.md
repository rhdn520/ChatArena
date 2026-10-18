# 테스트·검사기 정책 — 사용자 명시 지시, 2026-10-15

- 훅은 Slurm 자원·제출 규격 확인용으로만 사용한다.
- 그 외 테스트·검사기·preflight·smoke·모델 로딩 점검·응답 형식 검사·검증 게이트는 사용자의 명시 요청 없이 만들거나 추가·확대·실행하지 않는다.
- 실행·재실행·수정·확인 요청을 테스트나 검사기 제작·실행 허가로 확대 해석하지 않는다. 확인 요청에는 기존 결과와 로그를 읽어 답한다.
- 검사 필요성을 자의적으로 판단하여 작업을 지연하거나 실행을 차단하지 않는다. 삭제한 검사를 되살리거나 이름만 바꿔 재도입하지 않는다.
- 테스트 개수나 검사 통과를 실제 실행 성공 대신 보고하지 않는다. 관측한 실행 상태와 한계를 그대로 보고한다.
- 이 정책은 사용자 관리 지침에 남아 있던 자동 사전검사·스모크·실행 게이트 강제 규정을 대체한다. Slurm 규격 확인은 유지한다.
- 기존 실행환경·경로·모델·자원·로그를 보존한다. 사용자 지시 없는 폴링 금지.

# ChatArena Slurm settings

Use gpu:A6000:1, nodes=1, cpus-per-task=2, mem=32G, time=48:00:00.
No partition is explicitly set. Logs remain in ChatArena/logs and outputs in outputs/qwen35_pilot.
Use the existing interpreter, environment, model/cache and launcher.
The Slurm specification hook is the only automatic checker. It reads directives and resources only.
Submission is resource report and specification check, then sbatch. No model/generation preflight.
The job starts vLLM, waits for readiness and runs the games directly, retaining normal logs.
