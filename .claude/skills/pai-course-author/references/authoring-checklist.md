# 집필 검증 체크리스트 (Phase 4)

초안 생성 후 항목별로 실행합니다. 실패하면 `pai-agent`를 **해당 산출물만** 대상으로 재호출합니다.

---

## A. frontmatter

- [ ] 필수 필드 완비: `module` `week` `order` `title` `slug` `tier` `priority` `prereq` `tags` `est_reading_min` `updated` `sources_checked`
- [ ] `module`이 마스터플랜 모듈 ID와 정확히 일치 (`W1-M3` 형식)
- [ ] `order`가 폴더 숫자 prefix와 일치 (`03-diffusion-ddpm-dit` → `order: 3`)
- [ ] `slug`가 폴더명의 숫자 뒤 부분과 일치
- [ ] `tier`가 `docs/course-plan.md` §4 배정과 일치
- [ ] `sources_checked`가 Phase 2 확인 날짜

```bash
# 스키마 육안 확인
head -20 <target_dir>/lesson.md
```

## B. 링크·이미지

- [ ] 모든 상대링크가 실존 파일을 가리킴
- [ ] 이미지 참조는 `../../img/{module-id}-{slug}.png` 형태
- [ ] "다음 토픽" 링크가 다음 모듈 폴더를 정확히 가리킴

```bash
# lesson.md의 상대링크 추출 후 존재 확인
grep -oE '\]\(([^)]+)\)' <target_dir>/lesson.md
```

## C. 시각자료

- [ ] 최소 4종: ① 계보·지형 ② 아키텍처 ③ **회사 스택 연결도** ④ 비교표
- [ ] Mermaid가 ```` ```mermaid ```` 코드펜스 안에 있고 문법이 유효
- [ ] 아키텍처 블록도에 **입출력과 차원**이 명시됨
- [ ] 비교·트레이드오프·하이퍼파라미터는 산문이 아니라 표로

## D. 본문 필수 요소

- [ ] 학습 목표 3개 이상 + 완료 기준 1줄
- [ ] **회사 스택 연결 섹션 존재** (없으면 미완성)
- [ ] 흔한 오해 3가지
- [ ] 셀프 체크 퀴즈 10문항 + `<details>` 접힘 정답
- [ ] 출처에 arXiv ID·URL·확인 날짜
- [ ] 회사 내부 구현을 추측한 문장이 없는가 → 있으면 "팀 확인 필요"로 바꾸고 `notes/questions-for-team.md`에 적립
- [ ] 분량이 Tier 범위 안 (A 8~10장 / B 5~7장 / C 3~4장)

## E. practice 코드

- [ ] `requirements.txt` 버전 고정 (Phase 2에서 확인한 버전)
- [ ] `.py`에 jupytext percent 헤더와 `# %%` 셀 구분
- [ ] 핵심 수식↔코드 대응 주석 (`# eq.(3)`)
- [ ] 무거운 학습에 `--smoke` 플래그 존재
- [ ] **뷰어를 띄우는 코드가 없음** (`mujoco.viewer`, `plt.show()` 등) — 결과는 파일로 저장
- [ ] `MUJOCO_GL=egl` 전제가 README에 명시
- [ ] 출력 경로가 `artifacts/{module-id}/`

```bash
# 뷰어 코드 잔존 확인
grep -rnE 'viewer|plt\.show|cv2\.imshow' <target_dir>/practice/

# CPU 실행 검증
python <target_dir>/practice/NN_*.py --smoke
```

## F. jupytext 왕복

- [ ] `.py` ↔ `.ipynb` 동기화가 깨지지 않음
- [ ] 노트북 출력이 비워진 상태로 저장됨

```bash
jupytext --set-formats py:percent,ipynb <target_dir>/practice/NN_*.py   # 최초 1회
jupytext --sync <target_dir>/practice/NN_*.py
```

## G. labs

- [ ] 사전 준비 체크리스트(설치 명령 + 버전)
- [ ] **각 단계마다 명령 + 성공 판정 기준** — 시뮬 입문자에게 가장 중요
- [ ] 결과 해석 가이드 (어떤 커브·지표를 보고 무엇을 판단하는가)
- [ ] 흔한 에러 3~5개와 대처법
- [ ] 심화 변형 과제 2개

## H. 미검증 배지

- [ ] GPU가 필요해 실행하지 못한 절차마다 배지가 있음
- [ ] 배지에 예상 소요 시간과 인스턴스 타입이 적혀 있음
- [ ] 30분 이상 걸릴 학습은 예상 GPU 시간이 명시됨

```markdown
> ⚠️ **미검증(GPU 필요)** — 집필 시점에 실행 검증되지 않았습니다.
> 예상 소요: A10G 기준 약 40분. 실행 후 결과를 `docs/progress.md`에 기록하고 이 배지를 제거하세요.
```

---

## 통과 후

Phase 5(윤문)로 넘어갑니다. **윤문 전 상태를 커밋하거나 사본을 남겨** Phase 6 diff 대조에 쓰세요.
