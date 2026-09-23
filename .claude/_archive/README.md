# _archive — 이 저장소에서 쓰지 않는 스킬·에이전트 보관소

Claude Code는 `.claude/agents/`와 `.claude/skills/`만 탐색합니다.
여기 있는 것들은 **파일로는 남아 있지만 세션에 로드되지 않습니다.**

## 왜 옮겼나

에이전트 description은 지연 로딩이 아니라 **매 세션 시스템 프롬프트에 상주**합니다.
옮기기 전 에이전트 21종의 description 합계가 12.8KB였고 그중 8.9KB(69%)가
이 저장소와 무관한 Next.js·FastAPI·PRD 작업용이었습니다.

## 무엇이 있나

| 경로 | 내용 | 옮긴 이유 |
|---|---|---|
| `agents/dev/` | nextjs-app-developer, ui-markup-specialist, starter-cleaner, backend-developer, code-reviewer, development-planner | Next.js·FastAPI·ROADMAP.md 전용. 이 저장소에 해당 작업이 없음 |
| `agents/docs/` | prd-generator, prd-validator | PRD 작업 없음 |
| `agents/humanize/` | humanize-web-architect, taxonomy-gap-analyzer, translationese-research-distiller, korean-translation-scholar, post-editese-metric-engineer, quick-rules-integrator | 분류체계 유지보수 전용. `humanize-korean/SKILL.md`가 「필요 에이전트 6종」으로 명시한 실행 경로 밖 |
| `skills/prd`, `skills/prd-to-plan`, `skills/to-prd` | PRD 작성·분해 | PRD 작업 없음. 세 스킬이 같은 일을 중복 |
| `skills/frontend-design` | 웹 UI 디자인 | 마크다운 저작 저장소에 해당 없음 |
| `commands/docs/update-roadmap.md` | `/update-roadmap` 커맨드 | 이 저장소에 `ROADMAP.md`가 없다. allowed-tools가 없는 파일(`docs/ROADMAP.md`)을 가리키고 있었다 |

## 되돌리는 법

```bash
git mv .claude/_archive/agents/dev .claude/agents/dev        # 디렉토리 통째
git mv .claude/_archive/skills/prd .claude/skills/prd        # 스킬 하나
```

humanize 유지보수 에이전트가 필요한 회차(분류체계 v2 승격 등)에는 그때만 되돌리고,
작업이 끝나면 다시 여기로 옮깁니다.
