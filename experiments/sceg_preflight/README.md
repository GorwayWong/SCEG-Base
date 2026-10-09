# Full-episode resource preflight

Separate experiment: random 200-image 10-way5-shot15-query forward/backward,
zero optimizer updates. Config/log/checkpoint owned by each run. Checkpoint is
explicitly UNTRAINED and cannot be used as a baseline performance result.

Formal accuracy and confidence_interval remain null. This only verifies that the
released full episode fits the actual GPU and all expected active gradients are finite.
