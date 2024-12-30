import unittest
import torch
from src.models import SDFNetwork
from src.encoding import HashGridSDF
from src.data.datasets import SyntheticDataset

class TestSDF(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hash_encoder = HashGridSDF()
        self.model = SDFNetwork(self.hash_encoder).to(self.device)

    def test_hash_encoder(self):
        points = torch.rand((100, 3)).to(self.device)
        features = self.hash_encoder(points)
        self.assertEqual(features.shape, (100, self.hash_encoder.n_features))

    def test_sdf_prediction(self):
        points = torch.rand((100, 3)).to(self.device)
        sdf = self.model(points)
        self.assertEqual(sdf.shape, (100, 1))

    def test_synthetic_dataset(self):
        dataset = SyntheticDataset(num_points=1000)
        self.assertEqual(len(dataset), 1000)
        sample = dataset[0]
        self.assertIn('coords', sample)
        self.assertIn('sdf', sample)

if __name__ == '__main__':
    unittest.main()