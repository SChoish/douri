# 프로젝트 구조 점검 리포트 (2026-04-21)

## 전체 판단

- **최적화 수준: 중간(6.5/10)**
- 장점: 역할별 디렉터리(`agents/`, `utils/`, `config/`, `scripts/`, `docs/`)가 분리되어 있고, 실행 엔트리포인트가 기능별로 구분되어 있습니다.
- 한계: 엔트리포인트 간 중복, 일부 레거시 문서/엔트리 정합성 이슈, 대형 단일 파일(`utils/datasets.py`)로 인해 유지보수 비용이 높습니다.

## 근거 요약

1. 실행 엔트리포인트를 단일화해 운영 경로를 고정함
  - `main.py`
2. 단일 엔트리포인트 기준으로 문서/실행 경로 혼선이 크게 줄어듦
3. 파일 크기 관점에서 결합이 큰 모듈 존재
  - `utils/datasets.py`와 `agents/goub_dynamics.py`처럼 역할이 큰 파일에 로직이 집중됨
4. 정리 필요 신호
  - 문서가 예전 GOUB 파일명(`agents/goub_phase1.py`)을 가리키는 구간이 있었음
  - joint 학습은 `critic=deas|dqc` 선택과 별도 actor state를 함께 다루므로, README/설정 문서에서 이 경계를 계속 명확히 유지할 필요가 있음
  - chunk 길이 관련 용어는 `full_chunk` / `action_chunk` 기준으로 통일해 두는 편이 추후 유지보수에 유리함

## 우선순위별 정리 후보

### P0 (즉시)

- **문서-코드 기준선 일치**
  - README / 구조 리뷰 / 실행 스크립트가 현재 실제 파일명과 같은 기준을 보도록 유지할 필요가 있음.
  - 학습 진입점은 `main.py`, 구현 기준 파일은 `agents/goub_dynamics.py` + `agents/critic/`로 고정하는 편이 혼란이 적음.
  - critic-only / joint 모두 top-level `critic: deas | dqc` 선택을 공통 기준으로 보는 편이 혼란이 적음.

### P1 (단기)

- **엔트리포인트 공통화**
  - YAML 로딩, run dir 구성, 로깅/체크포인트, 시드 초기화 코드를 공용 유틸(예: `utils/train_runtime.py`)로 추출.
  - 기대효과: 버그 수정 시 중복 반영 방지, 신규 실험 엔트리 추가 속도 개선.

- **critic / actor 경계 유지**
  - 현재 joint 경로는 critic과 actor를 별도 agent/state/checkpoint로 분리해 두었으므로, 신규 실험을 추가해도 이 경계를 깨지 않는 편이 좋음.
  - 기대효과: DEAS/DQC 선택 로직 단순화, joint 디버깅 비용 감소.

- **데이터셋 모듈 분할**
  - `utils/datasets.py`를 `datasets/base.py`, `datasets/hgc.py`, `datasets/path_hgc.py` 등으로 분리.
  - 기대효과: 코드 검색성 향상, 테스트 단위 분리 용이.

### P2 (중기)

- **문서-코드 싱크 점검 자동화**
  - README에서 언급하는 스크립트/파일명 존재 여부를 검증하는 간단한 CI 스크립트 추가.

- **레거시 플래그 정리 계획**
  - `--checkpoint_step`, `--segment_len` 등 deprecated 경로의 제거 버전을 명시해 기술부채를 단계적으로 소거.

## 추천 실행 순서

1. 문서와 실제 엔트리포인트 기준선 동기화
2. Phase1 계열 엔트리포인트 공통 런타임 유틸 추출
3. 부분 지원 엔트리포인트 상태 표기
4. `utils/datasets.py` 분할
5. 문서-코드 싱크 CI 및 deprecated 제거 로드맵 추가

## 결론

현재 구조는 "실험을 빠르게 돌리기에는 충분히 실용적"이지만, 코드베이스가 커지는 시점에서 중복과 대형 파일이 병목이 될 가능성이 큽니다. 특히 엔트리포인트 공통화와 문서-코드 기준선 정리가 **효율 대비 효과가 가장 큰 1차 개선점**입니다.
