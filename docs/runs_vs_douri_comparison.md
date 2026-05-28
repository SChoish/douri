# Pathbridger runs vs douri baselines

Comparison between `/home/offrl/Pathbridger/runs` and matching configs under `/home/offrl/douri/runs`.

Metric is final eval success rate, shown as `IDM / Actor`.

- Pathbridger runs: epoch 600, 50 episodes per task.
- douri medium/large baselines: epoch 600, 50 episodes per task.
- douri giant baselines: original run is epoch 200; comparison below uses the final resume result at epoch 1000 with 50 episodes per task. The resume log also has epoch 600 with 10 episodes per task, so giant is not a perfectly epoch-matched comparison.

Mode labels:

- `rd`: residual target mode `displacement`
- `ra`: residual target mode `absolute`
- `sd`: subgoal target mode `displacement`
- `sa`: subgoal target mode `absolute`

## Baseline-Level Summary

| Group | Config | Best mode | douri run | douri IDM / Actor | Best Pathbridger IDM / Actor | Delta | Best Pathbridger run |
|---|---|---|---:|---:|---:|---:|---|
| giant-g2 | alpha=0.1, gap=5, kappa=0.6 | rd/sa | `20260517_234529` | 16 / 29 | 39 / 56 | +23 / +27 | `20260526_111944` |
| giant-g1 | alpha=0.5, gap=10, kappa=0.8 | rd/sa for IDM, ra/sa for Actor | `20260518_155648` | 40 / 40 | 38 / 42 | -2 / +2 | `20260525_120804` IDM, `20260525_234246` Actor |
| large-l1 | alpha=0.0, gap=10, kappa=0.9 | rd/sa or ra/sa for IDM, rd/sa for Actor | `20260520_073104` | 81 / 90 | 86 / 90 | +5 / 0 | `20260525_155343` or `20260525_181326` |
| large-l2 | alpha=0.0, gap=5, kappa=0.9 | ra/sa | `20260520_221741` | 72 / 83 | 83 / 85 | +11 / +2 | `20260526_043042` |
| medium-m1 | alpha=0.0, gap=5, kappa=0.7 | ra/sa or rd/sa for IDM, ra/sa, ra/sd, or rd/sa for Actor | `20260521_144335` | 94 / 94 | 94 / 96 | 0 / +2 | `20260525_202755`, `20260525_213239`, `20260525_223821` |
| medium-m2 | alpha=0.0, gap=5, kappa=0.6 | rd/sa for IDM, rd/sd or ra/sa for Actor | `20260521_133620` | 93 / 91 | 95 / 93 | +2 / +2 | `20260526_064546` IDM, `20260526_053950` or `20260526_085756` Actor |

## Pathbridger Variant Results

| Group | Mode | Residual target | Subgoal target | Pathbridger run | IDM / Actor |
|---|---:|---|---|---:|---:|
| giant-g2 | rd/sd | displacement | displacement | `20260526_100247` | 18 / 29 |
| giant-g2 | rd/sa | displacement | absolute | `20260526_111944` | 39 / 56 |
| giant-g2 | ra/sd | absolute | displacement | `20260526_123603` | 24 / 36 |
| giant-g2 | ra/sa | absolute | absolute | `20260526_135320` | 31 / 40 |
| giant-g1 | rd/sd | displacement | displacement | `20260525_105006` | 18 / 19 |
| giant-g1 | rd/sa | displacement | absolute | `20260525_120804` | 38 / 34 |
| giant-g1 | ra/sd | absolute | displacement | `20260525_132533` | 27 / 38 |
| giant-g1 | ra/sa | absolute | absolute | `20260525_234246` | 33 / 42 |
| large-l1 | rd/sd | displacement | displacement | `20260525_144325` | 79 / 81 |
| large-l1 | rd/sa | displacement | absolute | `20260525_155343` | 86 / 90 |
| large-l1 | ra/sd | absolute | displacement | `20260525_170215` | 83 / 82 |
| large-l1 | ra/sa | absolute | absolute | `20260525_181326` | 86 / 82 |
| large-l2 | rd/sd | displacement | displacement | `20260526_010030` | 79 / 81 |
| large-l2 | rd/sa | displacement | absolute | `20260526_021041` | 78 / 85 |
| large-l2 | ra/sd | absolute | displacement | `20260526_032012` | 78 / 70 |
| large-l2 | ra/sa | absolute | absolute | `20260526_043042` | 83 / 85 |
| medium-m1 | rd/sd | displacement | displacement | `20260525_192249` | 92 / 95 |
| medium-m1 | rd/sa | displacement | absolute | `20260525_202755` | 94 / 96 |
| medium-m1 | ra/sd | absolute | displacement | `20260525_213239` | 92 / 96 |
| medium-m1 | ra/sa | absolute | absolute | `20260525_223821` | 94 / 96 |
| medium-m2 | rd/sd | displacement | displacement | `20260526_053950` | 94 / 93 |
| medium-m2 | rd/sa | displacement | absolute | `20260526_064546` | 95 / 89 |
| medium-m2 | ra/sd | absolute | displacement | `20260526_075129` | 91 / 86 |
| medium-m2 | ra/sa | absolute | absolute | `20260526_085756` | 92 / 93 |

## Notes

- Medium is effectively saturated. The target-mode variants mostly stay around douri, with small Actor gains.
- Large improves on the `gap=5,kappa=0.9` douri baseline and matches or slightly improves the `gap=10,kappa=0.9` baseline.
- Giant-g2 has the clearest Pathbridger gain, especially `rd/sa` at `39 / 56`.
- Giant-g1 is mixed: Actor can edge out douri slightly, but IDM does not beat the douri final resume.
- For giant, compare with care because douri final resume is epoch 1000 while Pathbridger results are epoch 600.
