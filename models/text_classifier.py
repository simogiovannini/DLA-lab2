import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, input_shape: int, hidden_units_1: int, hidden_units_2: int, hidden_units_3: int, output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_1, out_features=hidden_units_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_2, out_features=hidden_units_3),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units_3, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
