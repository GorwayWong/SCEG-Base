"""Isolated DN4 configuration; all training and evaluation use the existing Dogs runner."""
from experiments.sceg_dogs_baseline.run import main
from .prepare import BASE
from .report import generate_report
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id")
    args, _ = parser.parse_known_args()
    try:
        exit_code = main(BASE/"configs/dogs.yaml")
    finally:
        if args.run_id and (BASE/"runs"/args.run_id/"metrics.json").exists():
            generate_report(args.run_id)
    raise SystemExit(exit_code)
