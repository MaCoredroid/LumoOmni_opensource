import json

from stage3_uti.tokenization.token_space import build_token_space_from_config


def test_token_space_build_and_mapping():
    cfg = {
        "base_text_model": "Qwen/Qwen3-8B-Base",
        "token_space": {
            "n_special": 8,
            "special_tokens": [
                "<|text_start|>",
                "<|text_end|>",
                "<|img_start|>",
                "<|img_end|>",
                "<|aud_start|>",
                "<|aud_end|>",
                "<|gen_text|>",
                "<|pad_mm|>",
            ],
        },
        "image_tokenizer": {"name_or_path": "dummy", "codebook_size": 16},
        "audio_codec": {"name_or_path": "dummy", "n_codebooks": 2, "codebook_size": 8},
    }

    ts = build_token_space_from_config(
        cfg,
        text_vocab_size=100,
        image_codebook_size=16,
        audio_codebook_size=8,
        n_codebooks=2,
    )
    ts.validate()

    assert ts.ranges["TEXT"]["start"] == 0
    assert ts.ranges["TEXT"]["size"] == 100

    img_global = ts.img_to_global(3)
    assert ts.img_from_global(img_global) == 3

    aud_global = ts.aud_to_global(1, 5)
    cb, tok = ts.aud_from_global(aud_global)
    assert cb == 1
    assert tok == 5

    payload = ts.to_json()
    assert payload["vocab_size_total"] == ts.vocab_size_total
    json.dumps(payload)
