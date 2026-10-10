# Official SCEG baseline

Fixed source commit: f7f130aeb5b1b49246150574c53a874d4ce8de80. CUB training completed 800 epochs, seed42; best epoch480. The user approved workers4 instead of official48; other numerical training parameters remain frozen in [configs/cub.yaml](configs/cub.yaml).

5-way 1/5-shot and 50-way 1-shot each completed 2000 episodes. **50-way 5-shot is paused by user request; the full baseline protocol is incomplete.** No tuning or retraining was performed.

Canonical [results.json](results/20261010_cub800_seed42_resume122/results.json), [report](../../docs/cub_reproduction_report.md), [table](../../docs/cub_result_table.md), [discrepancies](../../docs/cub_discrepancy.md).

Each runs/<run_id>/ records configuration, environment, command, source/data fingerprints, metrics and logs. The runtime baseline/evaluate/workflow scripts are frozen in the canonical result directory. Source modules are unchanged; only the recording adapter was adjusted to stop protecting obsolete output files after the requested cleanup.

Final last/best weights and RNG remain local; their SHA256 and full 800-epoch history remain auditable. All 40 validation metric records, best-epoch raw validation and completed test episode results are retained. No per-epoch weight archives. [Cleanup record](results/20261010_final_cleanup/cleanup_report.md). The unmeasured family-level result placeholder was removed; use the canonical per-run results.
