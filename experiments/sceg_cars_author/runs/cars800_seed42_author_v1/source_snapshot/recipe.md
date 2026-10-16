# Cars original SCEG: frozen local baseline

Experiment `sceg_cars_author`; protocol `sceg_author_cars_v1`; one formal run
`cars800_seed42_author_v1`. This is an author-class-split + public implementation +
explicit recipe, single-training-seed local baseline. Scores never select the recipe.

## Source and isolation

Worktree `/root/projects/aris/sceg_cars_author`, branch `experiment/sceg_cars_author`,
base `727a6b1` (the full hash is in each run's git_commit.txt). The requested reference
is `5ee680fa9fbcdd93fc3381a4d2e3aefe5fd52c82`. Between them, public code changes are
only trainer options for activation recording, epoch timing and deferred final evaluation;
other changes record the completed Dogs experiment. Those committed records are read-only.
This task neither imports a Dogs entry nor uses its results as a gate.

Official source is pinned to [Legenddddd/SCEG f7f130a](https://github.com/Legenddddd/SCEG/tree/f7f130aeb5b1b49246150574c53a874d4ce8de80).
The 13 requested source files were fetched from GitHub at that commit; the Cars parity
check uses a detached checkout at `/tmp/sceg-cars-official-reference`. That checkout is
only a test oracle, never an execution dependency of the formal run.

Paper: [CVPR 2026 main paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_From_Few-way_to_Many-way_Rethinking_Few-shot_Fine-grained_Image_Classification_CVPR_2026_paper.pdf),
§3.3, §3.4, §4.1.2; [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhao_From_Few-way_to_CVPR_2026_supplemental.pdf), §7.1–7.2.
Exa verified those sources; the main PDF was also available locally. Context7 was used
only to check API availability. Installed torch 2.6.0+cu124 / torchvision 0.21.0+cu124
remain unchanged, through `/root/projects/aris/sceg_fsfg/.venv/bin/python`.

## Recipe decisions

| Item | Paper | Active code / published configuration | This run and reason |
|---|---|---|---|
| Cars classes | 130/17/49 | author exact 1-based ID file | 130/17/49, sole membership source |
| intra | Cars 0.65 (§4.1.2) | parser default .85, overridable; run.sh covers CUB | .65 explicitly passed through supported loss config; no claim of an official Cars script |
| lamda | learnable-temperature description | parser 36; Cars 32 is commented; CUB script 38 | 36, generic code default; actual author Cars temperature remains unconfirmed |
| temperature optimization | learnable | optimizer contains model.parameters only; losses reconstructed each epoch | fixed; do not add loss parameters to optimizer |
| SC | Cars omits stage3→stage2 contribution | g_st executes, then output discarded for Cars | retain execution, BN updates and expected missing gradients |
| backbone / head | ResNet12, RGB84, three 640-d features | same; episode/global normalization differ | original formulas, residual FC, score sum, 130-class global head |
| training episode | episodic | 10-way, support5, query15, support-first; weighted depletion | same; no fixed episodes per epoch |
| objective | episode + global I²L | three episode terms + one global term | equal weights, inter .05, intra .65, lamda36 |
| optimizer | SGD .1, momentum .9, wd5e-4 | Nesterov explicitly enabled by published run.sh | SGD with those values, Nesterov=true |
| schedule | 800 epochs | epoch400 × stage2, StepLR400 gamma.1 | exactly 800, no early stop |
| validation | best validation model | every20, 5-way5-shot, 1000 episodes | same, only 17 validation classes; best by highest mean |
| test | 2000 episodes per setting | actual sampler fixes query16 | 5w1 → 5w5 → 49w1 → 49w5, 2000 each, same best |
| initialization / numeric | full original numeric environment unpublished | existing tested implementation | fresh seed42, FP32, AMP=false, TF32=false |
| workers / CPU | not confirmed | existing resource adaptation | workers4, CPU threads2, single GPU |

No active Cars-specific script contradicting these decisions was found in the pinned
release. This does not establish the authors' full original Cars training command.
No CUB/Dogs/old Cars checkpoint is loaded. No additional pretraining, ProKeR, augmentation,
loss weighting, clipping, accumulation or search is used.

## Data and labels

Author original: `data/provenance/sceg_author/StanfordCars_fewshot_splits.txt`, SHA256
`2677107210b2bf1c9dfe1741ddc7d6d8dad992232ff04efa24ba4676aa48052c`.
The existing source record describes author email supplied by the user; date and
publication permission remain unknown. No private message is published.

Stable raw copy: `/root/datasets/fasterVitBase/raw/cars/StanfordCars-Dataset-main/stanford_cars`.
Its README identifies the jhpohovey/StanfordCars GitHub mirror. `cars_train` (8144) and
`cars_test` (8041) are joined to their corresponding .mat annotations by `fname`.
`class` is the original 1-based ID; `cars_meta.class_names[ID-1]` gives its name.
The complete `cars_annos.mat` independently agrees in partition/class counts.
No assumption about an old ImageFolder label or DN4 membership is used.

Author membership produces **10747 / 1410 / 4028 images**. Destinations use three-digit
original ID folders and `cars_train__` / `cars_test__` filename prefixes. ImageFolder
sorts the training folders to labels 0..129; validation/test use their own contiguous
labels; episode labels are independently assigned 0..way-1. One full manifest is
`prep/samples.jsonl`; `prep/split.json` contains label maps, `prep/provenance.json`
contains source hashes/limitations, and `prep/prepared.json` records full decode checks.

Original files are linked without changing bytes. No boxes or parts are used.
The mirror has not been proven byte/pixel identical to Stanford's original archive;
local hashes identify this copy, not author pixel equivalence. All original images
were decoded during preparation; dimensions/modes are recorded in provenance.
Validation/test: RGB → Resize short edge92 → CenterCrop84 → PNG once; prepared loader
only ToTensor/Normalize. Training uses full images with official transform_type=0
online RandomResizedCrop84, ColorJitter(.4), horizontal flip, ToTensor/Normalize.

## Entry, state and evidence

From the isolated worktree, using the existing interpreter:

```bash
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.prepare
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m pytest -q experiments/sceg_cars_author/test_contracts.py
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.check_official --reference /tmp/sceg-cars-official-reference
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --mode check
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode pipeline
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode pipeline --resume
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode evaluate
rtk proxy /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_cars_author.run --run-id cars800_seed42_author_v1 --mode status
```

Initial training has no resume argument. Technical checks discard models and optimizers;
the new formal process sets seed42 and constructs all state from scratch. Resume accepts
only the same run/config/protocol/source latest at complete epoch boundaries. Eval-only
requires epoch800 and never trains. Complete runs are preserved without recomputation.
Only latest.pth and checkpoint.pth are retained; checkpoint is the validation best
(provisional before the first validation, never eligible for formal evaluation).

After training, the same best weights and training-end RNG/DropBlock counters feed the
four settings. Full raw results are validated and reused. Each missing setting starts
from its saved RNG boundary; partial settings can replay, never select another sample
for a better score. Indices are support-first, with sample paths and class IDs.
The default whole-episode evaluator is used. Any real 49-way memory limitation must
be recorded before introducing and validating a no-gradient encoding chunk path.

Raw accuracies are percentages; CI half-width is
`1.96 * sqrt(var(episode_accuracy, ddof=0) / 2000)`, recomputed on each report.
This is episode sampling uncertainty conditional on one model, not cross-seed stability.
The external status is training/evaluating/completed/failed (or waiting_for_gpu if blocked),
with separate training_completed/full_evaluation_completed. Missing scores are null.
Completed requires 800 epochs, all four verified results and the final report.

Run records include effective config, commands/cwd/environment, base commit/diff and a
snapshot of untracked Cars source (including nested configs), data fingerprints, epoch
curves, validation selection, two checkpoints, RNG boundaries and raw results.
The only new public modification separates train/validation wall time when the existing
optional timing flag is enabled. All other new code is in this Cars experiment.

Paper Table1 reference values: 5w1 91.18%, 5w5 97.35%, 49w1 64.87%, 49w5 84.08%.
These are context, never tuning or acceptance targets; local-paper differences are not
module gains. Future same-protocol comparisons should cite the completed run ID and
best SHA, only after this run actually completes. Stop after this single baseline.
