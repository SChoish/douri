# 프로젝트 구조 점검 리포트 (2026-04-21)

## 전체 판단

- **최적화 수준: 중간(6.5/10)**
- 장점: 역할별 디렉터리(`agents/`, `utils/`, `config/`, `scripts/`, `docs/`)가 분리되어 있고, 실행 엔트리포인트가 기능별로 구분되어 있습니다.
- 한계: 엔트리포인트 간 중복, 일부 네이밍/문서 불일치, 대형 단일 파일(`utils/datasets.py`)로 인해 유지보수 비용이 높습니다.

## 근거 요약

1. 실행 엔트리포인트가 다수 존재하며 목적별 분리가 되어 있음
   - `main_goub_phase1.py`, `main_goub_phase1_path.py`, `main_goub_phase2_policy.py`, `main_goub_chunk_low.py`, `main_deas_seq_critic.py`
2. 다만 `main_goub_phase1.py`와 `main_goub_phase1_path.py`는 구조가 매우 유사하여 공통 로직 추출 여지가 큼
   - 로컬 비교 기준 유사도 약 0.85
3. 파일 크기 관점에서 결합이 큰 모듈 존재
   - `utils/datasets.py` 592 lines, `agents/goub_phase1.py` 538 lines
4. 정리 필요 신호
   - 공백 포함 파일명: `agents/goub_phase1_ before_int_traj.py`
   - README의 스모크 테스트 파일명과 실제 파일명 간 불일치 가능성(`smoke_test_goub_phase1_path.py` vs `smoke_test_goub.py`)

## 우선순위별 정리 후보

### P0 (즉시)

- **파일명 정리/아카이브 정책 확정**
  - `agents/goub_phase1_ before_int_traj.py`는 임시 백업 파일로 보이며, import 실수/자동화 스크립트 누락/IDE 검색 노이즈를 유발할 수 있음.
  - 권장: `archive/`로 이동하거나 제거 기준 문서화.

### P1 (단기)

- **엔트리포인트 공통화**
  - YAML 로딩, run dir 구성, 로깅/체크포인트, 시드 초기화 코드를 공용 유틸(예: `utils/train_runtime.py`)로 추출.
  - 기대효과: 버그 수정 시 중복 반영 방지, 신규 실험 엔트리 추가 속도 개선.

- **데이터셋 모듈 분할**
  - `utils/datasets.py`를 `datasets/base.py`, `datasets/hgc.py`, `datasets/path_hgc.py` 등으로 분리.
  - 기대효과: 코드 검색성 향상, 테스트 단위 분리 용이.

### P2 (중기)

- **문서-코드 싱크 점검 자동화**
  - README에서 언급하는 스크립트/파일명 존재 여부를 검증하는 간단한 CI 스크립트 추가.

- **레거시 플래그 정리 계획**
  - `--checkpoint_step`, `--segment_len` 등 deprecated 경로의 제거 버전을 명시해 기술부채를 단계적으로 소거.

## 추천 실행 순서

1. 임시/백업 파일 정리 규칙 확정 및 적용
2. Phase1 계열 엔트리포인트 공통 런타임 유틸 추출
3. `utils/datasets.py` 분할
4. 문서-코드 싱크 CI 및 deprecated 제거 로드맵 추가

## 결론

현재 구조는 "실험을 빠르게 돌리기에는 충분히 실용적"이지만, 코드베이스가 커지는 시점에서 중복과 대형 파일이 병목이 될 가능성이 큽니다. 특히 엔트리포인트 공통화와 임시 파일 정리가 **효율 대비 효과가 가장 큰 1차 개선점**입니다.
