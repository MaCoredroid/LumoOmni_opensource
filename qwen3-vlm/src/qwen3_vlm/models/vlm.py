import torch
from torch import nn


class Qwen3VLM(nn.Module):
    def __init__(
        self,
        llm,
        vision,
        resampler,
        projector,
        image_patch_token_id,
        num_image_tokens,
        vision_ln=None,
    ):
        super().__init__()
        self.llm = llm
        self.vision = vision
        self.resampler = resampler
        self.projector = projector
        self.vision_ln = vision_ln
        self.image_patch_token_id = image_patch_token_id
        self.num_image_tokens = num_image_tokens

    def _encode_images(self, pixel_values):
        vision_out = self.vision(pixel_values=pixel_values)
        patches = vision_out.last_hidden_state
        if self.vision_ln is not None:
            patches = self.vision_ln(patches)
        resampled = self.resampler(patches)
        return self.projector(resampled)

    def build_inputs_embeds(self, input_ids, pixel_values=None, image_counts=None):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        if pixel_values is None or image_counts is None:
            return inputs_embeds

        projected = self._encode_images(pixel_values)
        total_images = int(sum(image_counts))
        if total_images != projected.size(0):
            raise ValueError(
                f"image batch mismatch: expected {total_images} images, got {projected.size(0)}"
            )

        image_idx = 0
        for batch_idx, num_images in enumerate(image_counts):
            num_images = int(num_images)
            if num_images == 0:
                continue
            num_tokens = num_images * self.num_image_tokens
            patch_positions = (input_ids[batch_idx] == self.image_patch_token_id).nonzero(as_tuple=False)
            patch_positions = patch_positions.flatten()
            if patch_positions.numel() != num_tokens:
                raise ValueError(
                    f"image token count mismatch: expected {num_tokens}, got {patch_positions.numel()}"
                )

            img_tokens = projected[image_idx : image_idx + num_images].reshape(num_tokens, -1)
            inputs_embeds[batch_idx, patch_positions] = img_tokens.to(inputs_embeds.dtype)
            image_idx += num_images

        return inputs_embeds

    def forward(self, input_ids, attention_mask, labels=None, pixel_values=None, image_counts=None):
        inputs_embeds = self.build_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_counts=image_counts,
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, pixel_values=None, image_counts=None, **kwargs):
        inputs_embeds = self.build_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_counts=image_counts,
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True
