import torch
import torch.nn as nn
import torch.nn.functional as F
from src.encoding import HashGridSDF

class SDFNetwork(nn.Module):
    def __init__(self, hash_encoder: HashGridSDF, hidden_dim=64):
        super().__init__()
        self.hash_encoder = hash_encoder
        input_dim = hash_encoder.n_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.hash_encoder(x)
        sdf = self.net(features)
        return sdf

    def get_sdf_gradient(self, x, epsilon=1e-4):
        with torch.enable_grad():
            x.requires_grad_(True)
            sdf = self.forward(x)
            grad = torch.autograd.grad(
                sdf.sum(), x, create_graph=True, retain_graph=True
            )[0]
        return grad