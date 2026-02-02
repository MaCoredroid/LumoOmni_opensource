import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from stage3_uti.tokenization.token_space import build_token_space_from_config
from stage3_uti.tokenization.unified_tokenizer import UnifiedTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uti-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.uti_config).read_text(encoding="utf-8"))
    text_tokenizer = AutoTokenizer.from_pretrained(cfg["base_text_model"])

    image_cfg = cfg.get("image_tokenizer", {})
    audio_cfg = cfg.get("audio_codec", {})
    if all(k in image_cfg for k in ("codebook_size",)) and all(
        k in audio_cfg for k in ("codebook_size", "n_codebooks")
    ):
        token_space = build_token_space_from_config(
            cfg,
            text_vocab_size=len(text_tokenizer),
            image_codebook_size=int(image_cfg["codebook_size"]),
            audio_codebook_size=int(audio_cfg["codebook_size"]),
            n_codebooks=int(audio_cfg["n_codebooks"]),
        )
    else:
        tokenizer = UnifiedTokenizer.from_config(args.uti_config, text_tokenizer=text_tokenizer)
        token_space = tokenizer.token_space
    token_space.validate()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    token_space.save_json(str(out_path))

    sha_path = out_path.with_suffix(".sha256")
    sha_path.write_text(token_space.sha256() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
