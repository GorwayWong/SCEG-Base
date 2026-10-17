# Fixed CUB-parameter Cars transfer

Status: completed. Fixed seed42 Cars model, best epoch 660; no training, Cars validation selection, parameter search, or extra seeds.
This experiment does not resolve the Cars paper reproduction gap and does not establish stability across training seeds.

| Setting | SCEG accuracy ± CI | Ridge accuracy ± CI | ProKeR accuracy ± CI | Ridge gain ± paired CI | ProKeR gain ± paired CI |
|---|---:|---:|---:|---:|---:|
| 5way_1shot | 90.252500 ± 0.333381 | 90.265000 ± 0.333119 | 90.256875 ± 0.333341 | +0.012500 ± 0.014064 | +0.004375 ± 0.008573 |
| 5way_5shot | 96.929375 ± 0.146079 | 97.115000 ± 0.138497 | 97.101875 ± 0.138907 | +0.185625 ± 0.064623 | +0.172500 ± 0.035205 |
| 49way_1shot | 62.198469 ± 0.121275 | 62.287883 ± 0.120171 | 62.284184 ± 0.120866 | +0.089413 ± 0.021225 | +0.085714 ± 0.012229 |
| 49way_5shot | 82.509503 ± 0.066274 | 81.843686 ± 0.064918 | 83.596875 ± 0.064276 | -0.665816 ± 0.051912 | +1.087372 ± 0.027100 |

Accuracy/CI are percentages; paired gain/CI are percentage points. Each CI is 1.96 × population std (ddof=0) / sqrt(2000). Paired intervals use per-episode method minus same-feature SCEG.

## Timing and resources

CUDA events are collected and synchronized once per 100 tasks. Transfer includes pageable CPU→GPU copy; CPU gather, label comparison/accuracy reduction, event bookkeeping and reporting are included only in group wall time. Shared preparation includes raw-support means, original prototype normalization, support/query kernel normalization and support one-hot labels. Each method interval includes support fitting (where applicable), query scoring and argmax; ProKeR computes its own f0, preserving its average-score scale. Add shared preparation to a method interval for its head latency. Backbone encoding and transfers are excluded from head latency. Event intervals may include CPU launch gaps; these are observed inference timings, not isolated solver benchmarks.

| Setting | Wall seconds | Transfer ms/task | Shared ms/task | SCEG ms/task | Ridge ms/task | ProKeR ms/task | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5way_1shot | 7.807 | 0.1928 | 0.5351 | 0.5422 | 0.6292 | 1.7349 | 0.098 |
| 5way_5shot | 7.934 | 0.3461 | 0.5360 | 0.5325 | 0.5848 | 1.6888 | 0.099 |
| 49way_1shot | 12.385 | 0.7445 | 0.5697 | 0.5501 | 1.2209 | 2.3053 | 0.125 |
| 49way_5shot | 13.257 | 0.8819 | 0.5573 | 0.5416 | 1.3258 | 2.3755 | 0.131 |

Encoding: 5.870 seconds; peak allocated 1.397 GiB; CPU cache 30935040 bytes, [4028,3,640] FP32. Batch/chunk 128, workers 4, CPU threads 2; AMP/TF32 off.
All-stage elapsed before CPU summary: 52.832 seconds. Short-check details and environment are in resolved.json.

## Necessary checks and history

Static loading validates every saved task index, support/query ordering, original class sequence and no repeated sample. Four episode-0 checks cover both shots and maximum shape; safe 128-chunk features and all method scores meet fixed FP32 tolerances with identical predictions. Original model.classify agrees with the migrated sum head; w=0 equals original score/3 exactly with identical predictions. BN buffers remain fixed; changing eval DropBlock counters leaves features exactly unchanged. No query labels enter fitting or normalization.

- 5way_1shot: 0 historical episode accuracy differences; mean impact +0.000000000 pp. Details and selective safe-replay evidence are retained in test_5way_1shot.json.
- 5way_5shot: 0 historical episode accuracy differences; mean impact +0.000000000 pp. Details and selective safe-replay evidence are retained in test_5way_5shot.json.
- 49way_1shot: 0 historical episode accuracy differences; mean impact +0.000000000 pp. Details and selective safe-replay evidence are retained in test_49way_1shot.json.
- 49way_5shot: 0 historical episode accuracy differences; mean impact +0.000000000 pp. Details and selective safe-replay evidence are retained in test_49way_5shot.json.

## Provenance and reproduction

Base: `88b4bf81a05e839770cf917e5188851eb53c2220`. ProKeR reference: `6facdc199acda7b17a2b3b6cda5cc353a39878c0`. Checkpoint SHA-256: `d4cf951237d05dbc751d9bc6196bd7fcc5d7cce9bed2c0b6dcbe0e9bca45168f`.
head.py carries the minimal reference algorithm. Differences are support-first reshape means, shared preparation, and validation at the loading boundary. Prototypes average raw supports before original normalization; f0 is the original stage3+stage4+stage2 sum divided by 3. A=solve(I+Kss/lambda,Y-f0(S)); f=f0+w*Kqs@A. Ridge uses solve(UsUs.T+lambda*I,Y), without intercept.
resolved.json contains the actual resolved Cars configuration, fixed parameters, commands, environment, input byte hashes, canonical parsed-config hash, base commit and executed source byte hashes. Historical indices/path maps are referenced, not duplicated. Existing data-preparation evidence is reused; images are not rehashed. Cache remains local. Source files are the actual added execution version and are not edited after launch.

Working directory: `/root/projects/aris/sceg_proker_cars`
```bash
/root/projects/aris/sceg_fsfg/.venv/bin/python -B -m experiments.sceg_proker_cars.run
/root/projects/aris/sceg_fsfg/.venv/bin/python -B -m experiments.sceg_proker_cars.run --summary
```
A completed run exits without rewriting. An incomplete run requires explicit continuation instructions; this entry does not auto-restart it.

Only experiments/sceg_proker_cars/ is added on experiment/sceg_proker_cars. Public modules, historical inputs and other worktrees are read-only; no commit/push/merge or environment changes. CPU-only reaggregation verified all 8000 records and saved statistics. No unresolved evaluation issues.
