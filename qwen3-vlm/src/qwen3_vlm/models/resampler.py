import torch
from torch import nn


class PerceiverResampler(nn.Module):
    def __init__(self, input_dim, num_latents, depth, num_heads, head_dim):
        super().__init__()
        latent_dim = num_heads * head_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        self.proj_in = nn.Linear(input_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(latent_dim),
                        "attn": nn.MultiheadAttention(latent_dim, num_heads, batch_first=True),
                        "ln2": nn.LayerNorm(latent_dim),
                        "ff": nn.Sequential(
                            nn.Linear(latent_dim, latent_dim * 4),
                            nn.GELU(),
                            nn.Linear(latent_dim * 4, latent_dim),
                        ),
                    }
                )
            )

    def forward(self, x):
        # x: (batch, seq, input_dim)
        x = self.proj_in(x)
        bsz = x.size(0)
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)

        for layer in self.layers:
            latents_norm = layer["ln1"](latents)
            attn_out, _ = layer["attn"](latents_norm, x, x, need_weights=False)
            latents = latents + attn_out
            latents = latents + layer["ff"](layer["ln2"](latents))

        return latents
