# Cars original SCEG — author split

Status: **completed**; epochs: 800/800.
Training completed: True; full evaluation completed: True.

Protocol: sceg_author_cars_v1. Classes 130/17/49; original images 10747/1410/4028 (16185 total).
Original annotation class IDs (1..196) map through author membership to zero-padded folders; ImageFolder train labels are 0..129; episode labels are 0..way-1. Original train/test filename prefixes prevent collisions.
Random initialization, seed42, FP32, no AMP/TF32; 4 workers, 2 CPU threads. Prepared train/val/test records remain read-only.
Best epoch: 660; best SHA256: d4cf951237d05dbc751d9bc6196bd7fcc5d7cce9bed2c0b6dcbe0e9bca45168f.

| Setting | Accuracy (%) | 95% CI half-width (pp) | Episodes | Seconds | Peak allocated bytes |
|---|---:|---:|---:|---:|---:|
| 5way_1shot | 90.252500 | 0.333381 | 2000 | 214.56 | 1191471616 |
| 5way_5shot | 96.929375 | 0.146079 | 2000 | 263.03 | 1410168832 |
| 49way_1shot | 62.198469 | 0.121275 | 2000 | 2068.88 | 1505639936 |
| 49way_5shot | 82.509503 | 0.066274 | 2000 | 2531.28 | 1507707904 |

49-way resource adaptation: FP32/no-grad encoding in chunks of 128, then the original full-episode classify. BN fixed; DropBlock counters advance once per logical episode. Training config, best and sampling protocol unchanged.
Numerical check: passed, two 5-way validation tasks; atol/rtol={'atol': 1e-05, 'rtol': 1e-05}. Evidence and source identity: evaluation_execution.json; exact changes: evaluation_source.patch.
Earlier slow run used about 15.22 GiB dedicated + 9.99 GiB shared GPU memory at the Windows VM level and briefly advanced about 34s/episode. This is not the same measurement as PyTorch peak allocated. Extra intermediate tensor residency is a contributor, not a proven complete explanation; the historical CUB result also argues against treating tensor retention alone as sufficient cause.
49way_1shot: chunk=128, total/2000=1.034s per episode; first episode (includes loader startup)=1.972s, first peak allocated=1.401GiB; full-setting peak allocated=1.402GiB.
49way_5shot: chunk=128, total/2000=1.266s per episode; first episode (includes loader startup)=2.122s, first peak allocated=1.403GiB; full-setting peak allocated=1.404GiB.

CI = 1.96 × sqrt(var(episode_accuracy, ddof=0) / 2000), in percentage points. All available summaries are recomputed from raw arrays.
One trained model: this CI describes episode sampling only, not variation across training seeds.

Recorded training + validation + checkpoint time: 38216.26s; peak train allocated GPU bytes: 10306869760.
Training compute/data time: 32857.50s; validation time: 5273.10s.
Interrupted partial epochs are not included in epoch totals; invocation starts, failures and resumes are in events.jsonl.

Recipe provenance: recipe.md. Cars intra=0.65 explicitly uses the paper value through the official parameter interface; fixed lambda36 is the generic parser default, not a confirmed Cars-specific training temperature. The Cars g_st gate still executes and updates BN but its output is discarded. No model formulas changed.
Run disk bytes at completion, before cleanup (including both checkpoints): 362716227. The subsequent cleanup removed 39 non-best validation detail files after checking all means/CIs against epoch summaries; epoch history, best validation raw data and all formal tests remain. See cleanup.json for exact deletion scope and retained-file hashes. Raw-image mirror equivalence to the original Stanford archive is unverified; full images were not cropped/re-encoded for training.
Original source: git_commit.txt + git_diff.patch + source_snapshot/ + execution_sources.json. Inference-only updates, when present: evaluation_source.patch + evaluation_execution.json. Original environment and training config remain preserved; resume_commands.txt records subsequent invocations.
Input fingerprints: data_check.json, referencing the original prep manifests without duplicating their sample list.
Raw tests bind the same best SHA and split SHA, with class names, sample indices, relative paths and episode accuracy; evaluation_boundaries.json preserves retries' RNG.
Implementation changes comprise the Cars entry/preparation, optional public trainer timing, and optional chunked evaluation; model formulas and training settings were preserved. CUB, Dogs, ProKeR and archives were not modified. No ProKeR gain is computed. This run is complete: the commands below are retained for reproducibility; continuation of a completed run returns without recomputation, and fresh training must use a different run ID.

Commands (cwd: /root/projects/aris/sceg_cars_author):

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode pipeline
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode pipeline --resume
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode evaluate
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode status
```
