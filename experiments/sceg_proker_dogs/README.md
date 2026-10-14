# Stanford Dogs: frozen SCEG + CUB-locked ProKeR

Completed run: [dogs_seed42_proker_transfer_v1](runs/dogs_seed42_proker_transfer_v1/REPORT.md).
Four settings × 2000 original Dogs episodes, query/class=16, seed42 checkpoint at epoch520.
No training or Dogs parameter search. Only this directory is added to the Dogs author branch.

## Inputs and implementation

- Dogs base: `727a6b1b3e9e60dd57cc0d2e7e4e014bb16abb6f` (`experiment/sceg_dogs_author`).
- CUB ProKeR source: `3b95006c2f0ebe71e19e64668a796fe6a65c6a8a` (`experiment/sceg_proker`).
- `head.py` is byte-identical to CUB's verified head; parameters and input paths are in [config.yaml](config.yaml).
- Data, checkpoint, original episode indices and sample maps are read-only references to the existing worktrees.
- Run `metadata.json` records exact executed source hashes. `source_snapshot/run.py` preserves that executed entry before the publication-only Git ancestry/audit fix; predictions, timings and recorded statistics are unchanged.

## Run and audit

Use the existing Python environment, from `/root/projects/aris/sceg_proker_dogs`:

```bash
# Completed run: exits without overwriting any artifact or running GPU evaluation.
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_proker_dogs.run --run-id dogs_seed42_proker_transfer_v1
# Read-only full input/source identity and CSV statistical audit; no GPU evaluation.
rtk proxy env CUDA_VISIBLE_DEVICES=0 /root/projects/aris/sceg_fsfg/.venv/bin/python -m experiments.sceg_proker_dogs.run --run-id dogs_seed42_proker_transfer_v1 --audit
```

A new run needs a new `--run-id`. For an incomplete run, `--resume` requires unchanged input and current execution-source identities. Auditing a completed run verifies the recorded execution source snapshot when the publication entry differs.

The cache-rename/metadata interruption window documented in the report fails closed if cache identity is missing. It did not occur in this run.

Git includes source, configuration, report, metadata, results, execution log, four episode-metrics CSVs and the historical-baseline replay CSV. Local `.gitattributes` preserves recorded run bytes, including the original CSV CRLF endings, so their SHA256 values survive checkout. The single 39.39 MiB feature cache, datasets, checkpoints, locks and bytecode remain local. Accuracy, normal CI and paired CI are reproducible from the committed CSVs without the feature cache; full input verification additionally needs the original worktrees and data paths.
