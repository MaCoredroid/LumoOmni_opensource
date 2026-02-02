import torch


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(precision):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32
