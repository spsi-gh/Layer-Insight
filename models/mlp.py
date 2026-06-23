import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim=4096):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1024)
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    return SimpleMLP()


def generate_inputs(args, device, dtype):
    return (
        torch.randn(
            args.batch_size,
            1024,
            device=device,
            dtype=dtype
        ),
    )