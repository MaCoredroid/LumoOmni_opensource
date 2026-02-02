from torch import nn


class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(output_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
