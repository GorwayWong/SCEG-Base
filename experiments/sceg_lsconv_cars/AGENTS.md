# Cars LSConv experiment

Only SC.self_s -> fixed LSConv; Cars seed42/800 epochs. See README.md and config.yaml.
Use model.build_model for training, resume and strict best evaluation.
Referenced baseline data, weights and results are read-only. No seed/recipe search.
Keep only best and rolling last; never edit frozen execution sources during a run.
One GPU-heavy process; detached launch, no AI polling or automatic retry.
