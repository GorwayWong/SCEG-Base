# Active Dogs project state

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

Cars now has an isolated author-protocol entry in this worktree: [recipe](sceg_cars_author/recipe.md), [configuration](sceg_cars_author/configs/cars.yaml), [data preparation](sceg_cars_author/prep/prepared.json), and [entry](sceg_cars_author/run.py). Original annotation IDs map to the author's 130/17/49 classes and 10747/1410/4028 images. The live Cars run state is [metrics](sceg_cars_author/runs/cars800_seed42_author_v1/metrics.json), with [report](sceg_cars_author/runs/cars800_seed42_author_v1/report.md); preparation or process launch does not imply completion. CUB, Dogs and ProKeR evidence are unchanged. Historical Dogs blocked/synthetic notes are not Cars dependencies.
