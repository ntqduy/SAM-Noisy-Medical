import argparse
from pathlib import Path

from runner.experiment import run_experiment
from runner.io_utils import load_yaml_config


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="configs/phase1.yaml or configs/phase2.yaml")
    ap.add_argument("--phase", type=int, choices=[1, 2], default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit_n", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(Path(args.config))

    if args.phase is not None:
        cfg["phase"] = int(args.phase)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.limit_n is not None:
        cfg["limit_n"] = int(args.limit_n)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
