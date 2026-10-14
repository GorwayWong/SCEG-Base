# Active Dogs project state

The active Dogs work is **original SCEG training and evaluation**, protocol `sceg_author_dogs_v1`. The completed preparation stage remains read-only. The seed42 run `dogs800_seed42_author_v1` was launched from random initialization on GPU0; its live state, not launch success, determines completion.

- Frozen recipe and source distinctions: [recipe.md](sceg_dogs_author/recipe.md).
- Independent configuration and entry: [config.yaml](sceg_dogs_author/config.yaml), [run.py](sceg_dogs_author/run.py).
- Live progress: [metrics.json](sceg_dogs_author/runs/dogs800_seed42_author_v1/metrics.json), [train.log](sceg_dogs_author/runs/dogs800_seed42_author_v1/train.log).
- Report, including same-run resume and evaluate-only commands: [REPORT.md](sceg_dogs_author/runs/dogs800_seed42_author_v1/REPORT.md). Missing formal measurements remain null; completed requires800 epochs and four verified2000-episode tests.

- Author originals and source semantics: [data/provenance/sceg_author/source.json](../data/provenance/sceg_author/source.json).
- Exact Dogs classes, counts and ImageFolder labels: [split.json](sceg_dogs_author/prep/split.json).
- Single image inventory: [samples.jsonl](sceg_dogs_author/prep/samples.jsonl).
- Data checks and preparation status: [prepared.json](sceg_dogs_author/prep/prepared.json).
- Stage handoff and actual commands: [handoff.md](sceg_dogs_author/prep/handoff.md).
- Archive and cleanup mappings: [cleanup_manifest.json](sceg_dogs_author/prep/cleanup_manifest.json).
- Superseded DN4 evidence: [archive README](_archive/sceg_dogs_dn4/dogs800_seed42_dn4/README.md).

The DN4 run remains evidence for its own public protocol. It is not an author-split reproduction: 62 class assignments differ and its training set includes 14 author test classes. Its weights must never initialize, resume, or formally evaluate the author protocol. The author entry passed its isolated contract tests and one real train/val model check before the separate random-initialized formal process started.

Cars is limited to preservation and validation of the author class-ID file. Its IDs are 1-based; Cars image mapping and image preparation have not been performed. CUB and ProKeR v1 are outside this closeout and remain unchanged. Historical Dogs blocked/synthetic notes and their shared test helpers are retained as history, not active split instructions or dependencies of the author preparation script.
