import time
import torch
from src.models import SDFNetwork, VanillaNetwork
from src.encoding import HashGridSDF, VanillaHashGrid
from src.data.datasets import SyntheticDataset

def benchmark_query_speed(model, dataset, device='cuda'):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    model = model.to(device)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            coords = batch['coords'].to(device)
            _ = model(coords)
    end_time = time.time()

    return end_time - start_time

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Synthetic dataset
    dataset = SyntheticDataset(num_points=100000, shape='sphere', radius=1.0)

    # SDF-HashNeRF
    sdf_hash_encoder = HashGridSDF()
    sdf_model = SDFNetwork(sdf_hash_encoder)
    sdf_time = benchmark_query_speed(sdf_model, dataset, device)

    # Vanilla HashNeRF
    vanilla_hash_encoder = VanillaHashGrid()
    vanilla_model = VanillaNetwork(vanilla_hash_encoder)
    vanilla_time = benchmark_query_speed(vanilla_model, dataset, device)

    print(f"SDF-HashNeRF Query Time: {sdf_time:.4f} seconds")
    print(f"Vanilla HashNeRF Query Time: {vanilla_time:.4f} seconds")

def benchmark_memory_usage(model, device='cuda'):
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        dummy_input = torch.rand((1024, 3), device=device)
        _ = model(dummy_input)
    return torch.cuda.max_memory_allocated(device)

if __name__ == "__main__":
    # Memory usage
    sdf_memory = benchmark_memory_usage(sdf_model, device)
    vanilla_memory = benchmark_memory_usage(vanilla_model, device)

    print(f"SDF-HashNeRF Memory Usage: {sdf_memory / 1e6:.2f} MB")
    print(f"Vanilla HashNeRF Memory Usage: {vanilla_memory / 1e6:.2f} MB")