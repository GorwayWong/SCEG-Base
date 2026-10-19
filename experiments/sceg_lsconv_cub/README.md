# SCEG × LSNet LSConv, CUB seed42

Single replacement: `enhanced_stage3 = F_s * gate_4_to_3 + LSConv(F_s)`;
then the original `t_s` / FC / three cosine classifiers / E/G constraints.
LSConv is the LSNet authors' module, migrated here; it is not a new module
invented by this experiment. The 7×7 LKP, 3×3 SKA, groups=8, GroupNorm,
BatchNorm, ReLU and internal residual retain upstream semantics/default initialization.
`c % 40` selects the dynamic kernel, not `c // 8`.

Source: https://github.com/THU-MIG/lsnet/tree/cbe737c92b7c43ecf02d08545a07f03f1010177c
Only Conv2d_BN/LKP/LSConv and model/ska.py are included. No timm, whole LSNet,
ImageNet weights, or new environment is required. No root license file exists
at the inspected upstream revision; no additional license is asserted here.

The complete original SCEG is initialized first. LSConv is then constructed
on CPU within `torch.random.fork_rng(devices=[])` and replaces `SC.self_s`.
All entrypoints use `build_model`; strict loading is mandatory. `enabled=False`
returns the original model without importing Triton or constructing extra parameters.

Run from `/root/projects/aris/sceg_lsconv_cub` using the existing interpreter:

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.check
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.run --mode train
# Start only after the training process has exited:
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.run --mode evaluate
# Same run and identity; only when interrupted with latest.pth available:
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cub.run --mode train --resume
```

Default run: `runs/20261019_cub800_seed42/`. One run_info.json holds effective
configuration, source snapshots including untracked new code, references,
precheck results, stages and resources. One metrics.jsonl records epoch metrics,
validation summaries and test progress. `checkpoint.pth` is validation best;
`latest.pth` is full continuation state saved every20 epochs and at normal end.
Atomic replacements require up to ~410MB for weights; precheck reserves 1GiB.
Four test JSONs reference exact baseline indices/path maps, contain new accuracies,
and share one best weight. `results.json` contains CPU-recomputed paired statistics.
Data and historical outputs stay in the original worktree and are read-only.

Completed run: [full report](runs/20261019_cub800_seed42/report.md) and
[paired statistics](runs/20261019_cub800_seed42/results.json). All 800 epochs
and four 2000-task tests completed. Gains for 5-way/1-shot, 5-way/5-shot,
50-way/1-shot and 50-way/5-shot are +0.279375, +0.260000, +0.229938 and
+0.498938 percentage points; all paired 95% intervals are above zero.
These results apply only to this CUB seed42 replacement.

Recipe: 800 epochs, seed42, 10-way/5-shot/15-query, original SGD/StepLR/I2L;
validation 5-way/5-shot, query16, 1000 tasks every20 epochs; four tests
(5/50-way × 1/5-shot), query16, 2000 tasks each. workers4, CPU threads2,
pin_memory=false, FP32, AMP/TF32 off. Encoding chunks128 are inference-only;
classification sees the whole task and DropBlock advances once per task.

Execution changes: no per-step gradient scan, activation dump, full data hash,
or duplicate training-end tests; detached epoch sums; last saved every20.
The original loss objects are recreated per epoch, lambda stays outside optimizer,
validation precedes scheduler.step, and best requires strict validation improvement.
No new numerical recipe, position/kernel/seed search, warmup, LR groups or ablation.
Report all four results including declines; single training seed supports no
cross-seed stability or attribution to an individual internal LSConv mechanism.
