import argparse

import yaml

from stage3_uti.utils.train_mmpt import train_mmpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_mmpt(cfg)


if __name__ == "__main__":
    main()
