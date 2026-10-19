# Cars fixed LSConv validation

Only `SC.self_s` is replaced. Cars `g_st` is still computed and discarded.
`model.build_model` constructs full Cars SCEG first, constructs LSConv inside
CPU `torch.random.fork_rng(devices=[])`, then returns the model for optimizer
creation or strict state loading. Disabled replacement consumes no extra RNG.

Base: Cars `88b4bf81a05e839770cf917e5188851eb53c2220`.
Module files copied verbatim from CUB `7539881285fd0e02c8ccb1c9142c9382e5e69c9b`.
LSNet by THU-MIG authors, upstream `cbe737c92b7c43ecf02d08545a07f03f1010177c`.
Original SCEG reference `f7f130aeb5b1b49246150574c53a874d4ce8de80`.
Keep upstream attribution; no new license assertion is made.

`config.yaml` references the completed Cars configuration. Effective config,
source hashes, uncommitted source snapshot and diff, checks and lifecycle state
live in one `run_info.json`; epoch/progress metrics in `metrics.jsonl`.
`checkpoint.pth` is strict-validation best; `latest.pth` is last every20/final.
Both retain full Cars checkpoint state, and explicit roles are checked on read.
Four formal JSONs reference historical task files without copying their arrays;
`results.json` is the independently recomputed CPU float64 paired summary.
No pretrained/CUB/baseline weight is loaded for initialization.

Use `/root/projects/aris/sceg_fsfg/.venv/bin/python` from this worktree:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cars.check
rtk proxy bash experiments/sceg_lsconv_cars/launch.sh
```

Launch the second command through nohup with stdout/stderr to the run's
`console.log`. It trains, waits for normal Python exit, evaluates in a new
process, and computes statistics in a third CPU process. There is no monitor,
retry or AI callback. Each phase's exit code is recorded in `console.log`.

Native recovery (only after inspecting a failure and confirming no active run):

```bash
rtk proxy bash experiments/sceg_lsconv_cars/launch.sh --resume
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cars.run --mode evaluate
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_lsconv_cars.run --mode statistics
```

Resume retains the original optimizer/scheduler/RNG/DropBlock state; at most
19 unsaved epochs may be repeated. Existing valid formal results are reused.
Frozen source or config conflicts fail explicitly. No automatic fallback.
One seed cannot establish cross-seed stability or mechanism causality.
Lambda36 is the completed Cars baseline's value, not an author-confirmed full
Cars recipe. Existing mirror provenance limitations remain. Uncontrolled wall
time differences are not evidence of an LSConv speedup.
