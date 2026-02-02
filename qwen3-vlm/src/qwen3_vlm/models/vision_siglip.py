from transformers import AutoImageProcessor, SiglipVisionModel


def build_vision_encoder(model_name, torch_dtype=None):
    model = SiglipVisionModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor
