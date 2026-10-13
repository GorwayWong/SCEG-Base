# Active Dogs project state

The active Dogs work is **author data preparation**, protocol `sceg_author_dogs_v1`. This stage prepares data only; it does not implement or start a training entrypoint.

- Author originals and source semantics: [data/provenance/sceg_author/source.json](../data/provenance/sceg_author/source.json).
- Exact Dogs classes, counts and ImageFolder labels: [split.json](sceg_dogs_author/prep/split.json).
- Single image inventory: [samples.jsonl](sceg_dogs_author/prep/samples.jsonl).
- Data checks and preparation status: [prepared.json](sceg_dogs_author/prep/prepared.json).
- Stage handoff and actual commands: [handoff.md](sceg_dogs_author/prep/handoff.md).
- Archive and cleanup mappings: [cleanup_manifest.json](sceg_dogs_author/prep/cleanup_manifest.json).
- Superseded DN4 evidence: [archive README](_archive/sceg_dogs_dn4/dogs800_seed42_dn4/README.md).

The DN4 run remains evidence for its own public protocol. It is not an author-split reproduction: 62 class assignments differ and its training set includes 14 author test classes. Its weights must never initialize, resume, or formally evaluate the author protocol. Future training must start from random initialization after a separate entrypoint/recipe/model-check stage.

Cars is limited to preservation and validation of the author class-ID file. Its IDs are 1-based; Cars image mapping and image preparation have not been performed. CUB and ProKeR v1 are outside this closeout and remain unchanged. Historical Dogs blocked/synthetic notes and their shared test helpers are retained as history, not active split instructions or dependencies of the author preparation script.
