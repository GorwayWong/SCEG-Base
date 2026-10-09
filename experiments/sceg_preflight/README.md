# Full-episode resource preflight

Separate experiment: random 200-image 10-way5-shot15-query forward/backward,
zero optimizer updates. Config/log/checkpoint owned by each run. Checkpoint is
explicitly UNTRAINED and cannot be used as a baseline performance result.

Formal accuracy and confidence_interval remain null. This only verifies that the
released full episode fits the actual GPU and all expected active gradients are finite.

The first working-tree preflight's checkpoint was byte-for-byte identical to
full_batch_200_committed/checkpoint.pth. Only that duplicate binary was removed;
checkpoint_reference.json retains the canonical relative path, byte size and SHA256.
Both runs' original configurations, code fingerprints, commands, metrics and logs remain.
