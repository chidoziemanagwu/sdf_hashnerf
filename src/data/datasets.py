import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDataset(Dataset):
    def __init__(self, num_points=100000, shape='sphere', radius=1.0):
        self.num_points = num_points
        self.shape = shape
        self.radius = radius
        self.points = torch.rand((num_points, 3)) * 2 - 1

        if shape == 'sphere':
            self.sdf = torch.norm(self.points, dim=-1) - radius
        elif shape == 'box':
            self.sdf = torch.max(torch.abs(self.points), dim=-1)[0] - radius
        else:
            raise NotImplementedError(f"Shape {shape} not implemented")

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return {'coords': self.points[idx], 'sdf': self.sdf[idx]}