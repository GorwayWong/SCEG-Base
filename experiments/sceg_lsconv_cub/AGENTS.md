# Scope

Only SC.self_s -> fixed upstream LSConv(320), CUB seed42/800 epochs; no recipe search.
Use model.build_model for training, resume and strict best-weight evaluation.
Read README.md for commands, sources, limits and output layout; config.yaml
references the verified baseline. Do not write to referenced baseline data/results.
One GPU-heavy process at a time; only evaluate after training exits.
Keep at most checkpoint.pth + latest.pth per run. Never call the old automatic
training-end tests in this experiment. No commits/pushes/merges or cleanup without request.
