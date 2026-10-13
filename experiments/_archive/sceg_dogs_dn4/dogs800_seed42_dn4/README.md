# Dogs DN4 historical evidence

Status: **superseded_by_author_split**. This archive preserves the original disk records of `dogs800_seed42_dn4`; source configs, test JSON, logs and historical report have not been rewritten. The task was already absent when the 2026-10-13 closeout inspected current processes, screen and CUDA compute owners; no termination signal was needed.

The original run completed 800 epochs with best epoch420. Disk records contain three completed settings, each 2000 episodes: 5-way1-shot, 5-way5-shot and 30-way1-shot. No complete 30-way5-shot record was present. It has not been evaluated again and its result remains missing/null. The historical `paused` status is a past snapshot, not a live process.

- Original run and two local weights: [run/](run/).
- Original historical report: [report.md](report.md).
- Original protocol/configuration evidence: [data_protocol/](data_protocol/) and [configs/](configs/).
- Author/DN4 class differences: [author_split_comparison.json](data_protocol/author_split_comparison.json).
- Migration/deletion/keep mappings and weight size/SHA256: [cleanup_manifest.json](../../../sceg_dogs_author/prep/cleanup_manifest.json).
- The single active author original location is [data/provenance/sceg_author/](../../../../data/provenance/sceg_author/).

Both `checkpoint.pth` and `latest.pth` are physically retained, one copy each, through same-filesystem rename. Moving them did not release disk space. Raw tests preserve episode accuracies, actual indices and their sample-path table. The original image package and archived protocol/preparation records permit reconstruction of old derived data; its redundant active image tree is retired after author data validation.

The 62 author/DN4 class assignment differences include **14 author test classes in the old training set**. These weights are forbidden as author-protocol training initialization, continuation, or formal evaluation. Future author-protocol training must use a new random initialization. These historical measurements remain valid as records of the stated DN4 public split, not author-split measurements.

Training source is traceable through the original run commit, git diff and untracked-source patch. Later committed helper/source files are recoverable from `56627bf7184f200a946c7378df789fd3e28e842c`; the repository is not copied into the archive. Original report links may retain historical paths; the cleanup mapping explains their new locations.
