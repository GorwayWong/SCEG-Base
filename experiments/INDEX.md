# SCEG author-split experiment state

## Cars: completed baseline

`cars800_seed42_author_v1` is **completed**: seed42 random initialization, 800 epochs, best validation epoch660, and four 2000-episode tests under `sceg_author_cars_v1`. The two 49-way settings use validated FP32 chunk128 encoding; weights, sampling protocol and completed 5-way results were preserved.

- [Final report and metrics](sceg_cars_author/runs/cars800_seed42_author_v1/report.md); [machine-readable report](sceg_cars_author/runs/cars800_seed42_author_v1/report.json).
- [Frozen recipe](sceg_cars_author/recipe.md), [configuration](sceg_cars_author/configs/cars.yaml), [entry](sceg_cars_author/run.py), [inference adaptation evidence](sceg_cars_author/runs/cars800_seed42_author_v1/evaluation_execution.json).
- [Data preparation and counts](sceg_cars_author/prep/prepared.json); [full sample mapping](sceg_cars_author/prep/samples.jsonl).
- [Cross-project performance audit](../docs/sceg_performance_audit_20261017.md); [cleanup and retained evidence](sceg_cars_author/runs/cars800_seed42_author_v1/cleanup.json).

Only non-best validation detail files, duplicate console output, a stale lock and regenerable Python caches were pruned. Epoch summaries, best validation details, all four raw test records, RNG boundaries, original source snapshots, best/latest weights and historical interruption evidence remain available. Weights and images remain local, outside Git.

## Dogs: existing completed baseline

The seed42 original SCEG author-split run **dogs800_seed42_author_v1 is completed**, protocol `sceg_author_dogs_v1`: 800 epochs and four verified 2000-episode formal evaluations. The prepared data and historical preparation handoff remain read-only.

- Frozen recipe and source distinctions: [recipe.md](sceg_dogs_author/recipe.md).
- Independent configuration and entry: [config.yaml](sceg_dogs_author/config.yaml), [run.py](sceg_dogs_author/run.py).
- Completed run records: [metrics.json](sceg_dogs_author/runs/dogs800_seed42_author_v1/metrics.json), [train.log](sceg_dogs_author/runs/dogs800_seed42_author_v1/train.log).
- Final report, paper comparison and recovery commands: [REPORT.md](sceg_dogs_author/runs/dogs800_seed42_author_v1/REPORT.md); independent result/count/identity checks: [final_checks.json](sceg_dogs_author/runs/dogs800_seed42_author_v1/final_checks.json). The original evaluation interruption remains unexplained; the missing setting completed after native continuation of its recovered process.

- Author originals and source semantics: [data/provenance/sceg_author/source.json](../data/provenance/sceg_author/source.json).
- Exact Dogs classes, counts and ImageFolder labels: [split.json](sceg_dogs_author/prep/split.json).
- Single image inventory: [samples.jsonl](sceg_dogs_author/prep/samples.jsonl).
- Data checks and preparation status: [prepared.json](sceg_dogs_author/prep/prepared.json).
- Stage handoff and actual commands: [handoff.md](sceg_dogs_author/prep/handoff.md).
- Archive and cleanup mappings: [cleanup_manifest.json](sceg_dogs_author/prep/cleanup_manifest.json).
- Superseded DN4 evidence: [archive README](_archive/sceg_dogs_dn4/dogs800_seed42_dn4/README.md).

The DN4 run remains evidence for its own public protocol. It is not an author-split reproduction: 62 class assignments differ and its training set includes 14 author test classes. Its weights must never initialize, resume, or formally evaluate the author protocol. The author entry passed its isolated contract tests and one real train/val model check before the separate random-initialized formal process started.

Cars uses the author's 130/17/49 classes and 10747/1410/4028 images; its completed evidence is indexed above. CUB, Dogs and ProKeR evidence are unchanged. Historical Dogs blocked/synthetic notes are not Cars dependencies.
