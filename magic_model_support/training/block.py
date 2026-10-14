"""Run one role's VERL block with full checkpoint state."""

import argparse
import json
import os
import signal
from pathlib import Path

from magic_model_support.training.config import compose_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=["attacker", "defender"])
    parser.add_argument("--target-step", required=True, type=int)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    os.environ["MAGIC_RUN_CONFIG"] = str(args.config.resolve())
    os.environ["MAGIC_TRAIN_ROLE"] = args.role
    config = compose_config(json.loads(args.config.read_text()), args.role, args.target_step)
    import ray
    from verl.trainer.main_ppo import main as train
    try:
        train(config)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
