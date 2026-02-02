from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_lora(model, lora_cfg):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(lora_cfg.get("target_modules", [])),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_lora(model, lora_path):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, lora_path)


def is_lora_model(model):
    return hasattr(model, "peft_config")


def build_llm(model_name, torch_dtype=None, attn_implementation=None, tokenizer_name=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    load_kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer
