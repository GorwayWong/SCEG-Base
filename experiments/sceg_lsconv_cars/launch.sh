#!/usr/bin/env bash
# One detached pipeline, separate Python processes; no retries or monitor.
set -u
cd /root/projects/aris/sceg_lsconv_cars || exit 1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
python_bin=/root/projects/aris/sceg_fsfg/.venv/bin/python
run_id=20261019_cars800_seed42
run_path=experiments/sceg_lsconv_cars/runs/$run_id
printf 'pipeline_pid=%s\n' "$$"
rtk proxy "$python_bin" -m experiments.sceg_lsconv_cars.run --mode train --run-id "$run_id" "$@"
code=$?
printf 'train_exit_code=%s\n' "$code"
if [ "$code" -ne 0 ]; then exit "$code"; fi
rtk proxy "$python_bin" -m experiments.sceg_lsconv_cars.run --mode evaluate --run-id "$run_id"
code=$?
printf 'evaluate_exit_code=%s\n' "$code"
if [ "$code" -ne 0 ]; then exit "$code"; fi
rtk proxy "$python_bin" -m experiments.sceg_lsconv_cars.run --mode statistics --run-id "$run_id"
code=$?
printf 'statistics_exit_code=%s\n' "$code"
exit "$code"
