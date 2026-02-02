import argparse

from qwen3_vlm.train.train_utils import train
from qwen3_vlm.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
