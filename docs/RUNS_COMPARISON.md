# runs/ 실험 결과 비교

> IDE에서 안 열리면 **`docs/RUNS_COMPARISON.md`** 를 여세요.

갱신: 2026-05-20 10:54 · grid CSV **56** configs · phase2 **13** configs

---

## Summary

**지금 (nohup):** 3×3 **α=0, gap>0** 기존 run → **600ep** resume (10/15 완료 · **11/15** `puzzle_3x3_a0p0_gap20p0_k0p9.yaml`) → 끝나면 4×4 **α=0, gap>0** 16개 (200→600)

| 항목 | 값 |
|------|-----|
| Grid 기록 | **56** configs (3×3 FBR displacement grid) |
| Best **Actor@400** | **20%** — α=0.0, gap=20.0, κ=0.6 |
| Best **IDM@400** | **21%** — α=0.1, gap=5.0, κ=0.6 |
| α=0 @600 | **10** / 15 (resume job, gap>0) |
| 제외 | gap=0 · 마지막 eval IDM·Actor 모두 0% |
| 학습 | stage1 200ep → (metric↑) 400ep · resume job은 **600ep** |

**α별 config 수:** α=0.0: 20 · α=0.1: 20 · α=0.3: 16

---

## Best (@400 eval, 50 ep/task)

| 지표 | α | gap | κ | 성능 | config |
|------|---|-----|---|------|--------|
| Actor | 0.0 | 20.0 | 0.6 | **20%** | `puzzle_3x3_a0p0_gap20p0_k0p6.yaml` |
| IDM | 0.1 | 5.0 | 0.6 | **21%** | `puzzle_3x3_a0p1_gap5p0_k0p6.yaml` |

---

## Config 비교표 (Actor@400 ↓)

| # | α | gap | κ | 400 | IDM@200 | Actor@200 | IDM@400 | **Actor@400** | config |
|---|--:|----:|--:|:---|---------|-----------|---------|---------------|--------|
| 1 | 0.0 | 20.0 | 0.6 | ✓ | 1% | 17% | 5% | **20%** | `puzzle_3x3_a0p0_gap20p0_k0p6.yaml` |
| 2 | 0.0 | 20.0 | 0.8 | ✓ | 11% | 20% | 10% | **20%** | `puzzle_3x3_a0p0_gap20p0_k0p8.yaml` |
| 3 | 0.0 | 5.0 | 0.7 | ✓ | 6% | 0% | 16% | **18%** | `puzzle_3x3_a0p0_gap5p0_k0p7.yaml` |
| 4 | 0.3 | 10.0 | 0.9 | · | 10% | 18% | - | **18%†** | `puzzle_3x3_a0p3_gap10p0_k0p9.yaml` |
| 5 | 0.3 | 20.0 | 0.8 | · | 9% | 18% | - | **18%†** | `puzzle_3x3_a0p3_gap20p0_k0p8.yaml` |
| 6 | 0.1 | 20.0 | 0.8 | ✓ | 10% | 19% | 8% | **17%** | `puzzle_3x3_a0p1_gap20p0_k0p8.yaml` |
| 7 | 0.0 | 10.0 | 0.6 | ✓ | 12% | 17% | 13% | **16%** | `puzzle_3x3_a0p0_gap10p0_k0p6.yaml` |
| 8 | 0.0 | 10.0 | 0.7 | ✓ | 4% | 1% | 0% | **16%** | `puzzle_3x3_a0p0_gap10p0_k0p7.yaml` |
| 9 | 0.0 | 5.0 | 0.8 | ✓ | 10% | 0% | 13% | **16%** | `puzzle_3x3_a0p0_gap5p0_k0p8.yaml` |
| 10 | 0.1 | 10.0 | 0.6 | ✓ | 15% | 8% | 10% | **16%** | `puzzle_3x3_a0p1_gap10p0_k0p6.yaml` |
| 11 | 0.1 | 10.0 | 0.7 | · | 16% | 16% | - | **16%†** | `puzzle_3x3_a0p1_gap10p0_k0p7.yaml` |
| 12 | 0.1 | 10.0 | 0.8 | ✓ | 11% | 13% | 13% | **16%** | `puzzle_3x3_a0p1_gap10p0_k0p8.yaml` |
| 13 | 0.1 | 10.0 | 0.9 | ✓ | 6% | 14% | 14% | **16%** | `puzzle_3x3_a0p1_gap10p0_k0p9.yaml` |
| 14 | 0.1 | 5.0 | 0.7 | ✓ | 7% | 0% | 9% | **16%** | `puzzle_3x3_a0p1_gap5p0_k0p7.yaml` |
| 15 | 0.0 | 10.0 | 0.8 | ✓ | 10% | 9% | 10% | **14%** | `puzzle_3x3_a0p0_gap10p0_k0p8.yaml` |
| 16 | 0.0 | 1.0 | 0.6 | ✓ | 4% | 8% | 8% | **14%** | `puzzle_3x3_a0p0_gap1p0_k0p6.yaml` |
| 17 | 0.3 | 5.0 | 0.8 | · | 4% | 14% | - | **14%†** | `puzzle_3x3_a0p3_gap5p0_k0p8.yaml` |
| 18 | 0.0 | 20.0 | 0.9 | · | 3% | 12% | - | **12%†** | `puzzle_3x3_a0p0_gap20p0_k0p9.yaml` |
| 19 | 0.1 | 1.0 | 0.8 | ✓ | 2% | 0% | 10% | **12%** | `puzzle_3x3_a0p1_gap1p0_k0p8.yaml` |
| 20 | 0.1 | 20.0 | 0.6 | ✓ | 2% | 4% | 17% | **12%** | `puzzle_3x3_a0p1_gap20p0_k0p6.yaml` |
| 21 | 0.3 | 10.0 | 0.6 | · | 16% | 12% | - | **12%†** | `puzzle_3x3_a0p3_gap10p0_k0p6.yaml` |
| 22 | 0.1 | 0.0 | 0.7 | ✓ | 0% | 0% | 2% | **11%** | `puzzle_3x3_a0p1_gap0p0_k0p7.yaml` |
| 23 | 0.3 | 0.0 | 0.7 | ✓ | 2% | 6% | 4% | **10%** | `puzzle_3x3_a0p3_gap0p0_k0p7.yaml` |
| 24 | 0.3 | 20.0 | 0.6 | · | 3% | 10% | - | **10%†** | `puzzle_3x3_a0p3_gap20p0_k0p6.yaml` |
| 25 | 0.1 | 1.0 | 0.7 | ✓ | 2% | 0% | 7% | **9%** | `puzzle_3x3_a0p1_gap1p0_k0p7.yaml` |
| 26 | 0.0 | 5.0 | 0.6 | ✓ | 12% | 5% | 10% | **7%** | `puzzle_3x3_a0p0_gap5p0_k0p6.yaml` |
| 27 | 0.0 | 5.0 | 0.9 | ✓ | 8% | 4% | 8% | **7%** | `puzzle_3x3_a0p0_gap5p0_k0p9.yaml` |
| 28 | 0.3 | 0.0 | 0.6 | ✓ | 0% | 0% | 8% | **7%** | `puzzle_3x3_a0p3_gap0p0_k0p6.yaml` |
| 29 | 0.3 | 5.0 | 0.6 | · | 3% | 7% | - | **7%†** | `puzzle_3x3_a0p3_gap5p0_k0p6.yaml` |
| 30 | 0.0 | 0.0 | 0.7 | ✓ | 12% | 0% | 7% | **6%** | `puzzle_3x3_a0p0_gap0p0_k0p7.yaml` |
| 31 | 0.1 | 1.0 | 0.6 | ✓ | 4% | 0% | 10% | **6%** | `puzzle_3x3_a0p1_gap1p0_k0p6.yaml` |
| 32 | 0.0 | 0.0 | 0.8 | ✓ | 2% | 0% | 4% | **5%** | `puzzle_3x3_a0p0_gap0p0_k0p8.yaml` |
| 33 | 0.1 | 5.0 | 0.6 | ✓ | 7% | 6% | 21% | **5%** | `puzzle_3x3_a0p1_gap5p0_k0p6.yaml` |
| 34 | 0.0 | 10.0 | 0.9 | · | 6% | 4% | - | **4%†** | `puzzle_3x3_a0p0_gap10p0_k0p9.yaml` |
| 35 | 0.0 | 1.0 | 0.9 | ✓ | 11% | 1% | 11% | **4%** | `puzzle_3x3_a0p0_gap1p0_k0p9.yaml` |
| 36 | 0.1 | 1.0 | 0.9 | ✓ | 2% | 0% | 4% | **4%** | `puzzle_3x3_a0p1_gap1p0_k0p9.yaml` |
| 37 | 0.1 | 20.0 | 0.9 | · | 2% | 4% | - | **4%†** | `puzzle_3x3_a0p1_gap20p0_k0p9.yaml` |
| 38 | 0.1 | 5.0 | 0.8 | ✓ | 1% | 0% | 8% | **4%** | `puzzle_3x3_a0p1_gap5p0_k0p8.yaml` |
| 39 | 0.1 | 0.0 | 0.8 | ✓ | 7% | 10% | 4% | **3%** | `puzzle_3x3_a0p1_gap0p0_k0p8.yaml` |
| 40 | 0.3 | 5.0 | 0.7 | · | 10% | 3% | - | **3%†** | `puzzle_3x3_a0p3_gap5p0_k0p7.yaml` |
| 41 | 0.0 | 0.0 | 0.9 | · | 6% | 2% | - | **2%†** | `puzzle_3x3_a0p0_gap0p0_k0p9.yaml` |
| 42 | 0.0 | 1.0 | 0.7 | ✓ | 4% | 0% | 9% | **2%** | `puzzle_3x3_a0p0_gap1p0_k0p7.yaml` |
| 43 | 0.1 | 0.0 | 0.6 | ✓ | 6% | 1% | 8% | **2%** | `puzzle_3x3_a0p1_gap0p0_k0p6.yaml` |
| 44 | 0.0 | 0.0 | 0.6 | ✓ | 0% | 0% | 7% | **1%** | `puzzle_3x3_a0p0_gap0p0_k0p6.yaml` |
| 45 | 0.1 | 0.0 | 0.9 | ✓ | 0% | 0% | 3% | **1%** | `puzzle_3x3_a0p1_gap0p0_k0p9.yaml` |
| 46 | 0.3 | 0.0 | 0.8 | ✓ | 4% | 1% | 6% | **1%** | `puzzle_3x3_a0p3_gap0p0_k0p8.yaml` |
| 47 | 0.3 | 10.0 | 0.8 | · | 0% | 1% | - | **1%†** | `puzzle_3x3_a0p3_gap10p0_k0p8.yaml` |
| 48 | 0.0 | 1.0 | 0.8 | · | 4% | 0% | - | **0%†** | `puzzle_3x3_a0p0_gap1p0_k0p8.yaml` |
| 49 | 0.0 | 20.0 | 0.7 | ✓ | 0% | 0% | 0% | **0%** | `puzzle_3x3_a0p0_gap20p0_k0p7.yaml` |
| 50 | 0.1 | 20.0 | 0.7 | ✓ | 0% | 0% | 2% | **0%** | `puzzle_3x3_a0p1_gap20p0_k0p7.yaml` |
| 51 | 0.1 | 5.0 | 0.9 | · | 4% | 0% | - | **0%†** | `puzzle_3x3_a0p1_gap5p0_k0p9.yaml` |
| 52 | 0.3 | 0.0 | 0.9 | · | 3% | 0% | - | **0%†** | `puzzle_3x3_a0p3_gap0p0_k0p9.yaml` |
| 53 | 0.3 | 10.0 | 0.7 | · | 0% | 0% | - | **0%†** | `puzzle_3x3_a0p3_gap10p0_k0p7.yaml` |
| 54 | 0.3 | 20.0 | 0.7 | · | 0% | 0% | - | **0%†** | `puzzle_3x3_a0p3_gap20p0_k0p7.yaml` |
| 55 | 0.3 | 20.0 | 0.9 | · | 0% | 0% | - | **0%†** | `puzzle_3x3_a0p3_gap20p0_k0p9.yaml` |
| 56 | 0.3 | 5.0 | 0.9 | · | 4% | 0% | - | **0%†** | `puzzle_3x3_a0p3_gap5p0_k0p9.yaml` |

† = 400ep 미진행, @200 best

---

## Top 5 Actor @400

1. **20%** — α=0.0, gap=20.0, κ=0.6 · `puzzle_3x3_a0p0_gap20p0_k0p6.yaml`
2. **20%** — α=0.0, gap=20.0, κ=0.8 · `puzzle_3x3_a0p0_gap20p0_k0p8.yaml`
3. **18%** — α=0.0, gap=5.0, κ=0.7 · `puzzle_3x3_a0p0_gap5p0_k0p7.yaml`
4. **18%** — α=0.3, gap=10.0, κ=0.9 · `puzzle_3x3_a0p3_gap10p0_k0p9.yaml`
5. **18%** — α=0.3, gap=20.0, κ=0.8 · `puzzle_3x3_a0p3_gap20p0_k0p8.yaml`

---

## IDM heatmap — **α=0 resume → 600ep** (신규)

**3×3 · 15 configs** (gap>0, 마지막 eval 0% 제외) · epoch **600** eval IDM

| | 파일 |
|--|------|
| PNG | [`idm_heatmap_alpha0_epoch600_3x3.png`](figures/idm_heatmap_alpha0_epoch600_3x3.png) |
| SVG | `docs/figures/idm_heatmap_alpha0_epoch600_3x3.svg` |

최고: **gap=10, κ=0.8 → 19%**

**4×4 α=0 @600 (진행 중 10/16):** [`idm_heatmap_alpha0_epoch600_4x4.png`](figures/idm_heatmap_alpha0_epoch600_4x4.png) — 지금까지 전부 IDM **0%**

---

## IDM heatmap — 전체 grid (@200, 참고)

**57 configs** · epoch **200** IDM · α=0/0.1/0.3/0.5

[`idm_heatmap_all_alpha.png`](figures/idm_heatmap_all_alpha.png) · [`idm_heatmaps.html`](figures/idm_heatmaps.html)

---

## 로그

- `puzzle_grid.master.log` · `alpha0_resume600.master.log`
- `sweep_results/puzzle_fbr_displacement_grid.csv`

*자동 생성*
