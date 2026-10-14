# Dogs original SCEG — author split

Status: **running**; epochs: 800/800.
Training completed: True; full evaluation completed: False.

Protocol: sceg_author_dogs_v1. Classes 70/20/30; original images 11877/3325/5378 (20580 total).
Random initialization, seed42, FP32, no AMP/TF32; 4 workers, 2 CPU threads. Prepared train/val/test records remain read-only.
Best epoch: 520; best SHA256: 1939b5e0513b88702dcc06a9413f67c14e247cea1d4be0eb51b2c3ba1810b7bb.

| Setting | Accuracy (%) | 95% CI half-width (pp) | Episodes | Seconds | Peak allocated bytes |
|---|---:|---:|---:|---:|---:|
| 5way_1shot | 77.164375 | 0.482614 | 2000 | 215.56 | 1126520832 |
| 5way_5shot | 88.740000 | 0.256899 | 2000 | 264.07 | 1345182720 |
| 30way_1shot | 47.075625 | 0.171086 | 2000 | 17290.95 | 5768955904 |
| 30way_5shot | null | null | null | null | null |

CI = 1.96 × sqrt(var(episode_accuracy, ddof=0) / 2000), in percentage points. All available summaries are recomputed from raw arrays.
One trained model: this CI describes episode sampling only, not variation across training seeds.

Recorded training + validation + checkpoint time: 46060.10s; peak train allocated GPU bytes: 10383925248.
Interrupted partial epochs are not included in epoch totals; invocation starts, failures and resumes are in events.jsonl.

Recipe provenance and paper reference numbers: recipe.md in the recorded source snapshot. Public CUB run.sh supplies lambda38; Dogs-specific complete recipe remains unconfirmed. Code keeps fixed lambda despite the paper's learnable-temperature description.
Source: git_commit.txt + git_diff.patch for tracked files, source_snapshot/ for new entry files, execution_sources.json for loaded source identity; environment.json and command.txt record the actual invocation.
Input fingerprints: data_check.json, referencing the original prep manifests without duplicating their sample list.
Raw tests bind the same best SHA and split SHA, with class names, sample indices, relative paths and episode accuracy; evaluation_boundaries.json preserves retries' RNG.
This task changes only the author entry and optional public trainer recording/finalization. CUB, ProKeR and archives are not written. No ProKeR gain is computed.

Commands (cwd: /root/projects/aris/sceg_dogs_baseline):

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode pipeline
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode pipeline --resume
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode evaluate
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_dogs_author.run --run-id dogs800_seed42_author_v1 --mode status
```
