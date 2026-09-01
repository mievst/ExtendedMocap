import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """Dense regressor that maps MediaPipe feature vectors to bone rotation
    quaternions. Mirrors the architecture used in the original training
    notebooks (mediapipe2quatDL.ipynb).
    """

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.5) -> None:
        super().__init__()
        self.sec = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sec(x)
