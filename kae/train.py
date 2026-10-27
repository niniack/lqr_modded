"""Train with a TOML recipe; --resume continues output_dir/resume.pt without resetting epochs."""

import argparse

from kae.training import Trainer, TrainingConfig
from kae.training.checkpointing import PAUSED_EXIT_CODE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TOML training recipe")
    parser.add_argument("--no-wandb", action="store_true", help="Disable remote logging")
    parser.add_argument("--resume", action="store_true",
                        help="Continue output_dir/resume.pt with the exact same recipe")
    args = parser.parse_args()
    config = TrainingConfig.from_toml(args.config)
    if args.no_wandb:
        config.wandb = False
    trainer = Trainer(config)
    trainer.run(resume=args.resume)
    if trainer.paused:
        raise SystemExit(PAUSED_EXIT_CODE)


if __name__ == "__main__":
    main()
