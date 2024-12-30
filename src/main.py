import torch
from torch.utils.data import DataLoader
from src.models import SDFNetwork
from src.encoding import HashGridSDF
from src.data.datasets import SyntheticDataset
from src.training import train_sdf_model
from .utils.visualization import visualize_sdf_slice

def main():
    # Configuration
    config = {
        "num_points": 100000,
        "shape": "sphere",
        "radius": 1.0,
        "batch_size": 1024,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "checkpoints"
    }

    # Initialize dataset and dataloader
    dataset = SyntheticDataset(
        num_points=config["num_points"],
        shape=config["shape"],
        radius=config["radius"]
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Initialize model
    hash_encoder = HashGridSDF()
    model = SDFNetwork(hash_encoder)

    # Train the model
    trained_model = train_sdf_model(
        model=model,
        train_loader=dataloader,
        num_epochs=config["num_epochs"],
        lr=config["learning_rate"],
        device=config["device"],
        checkpoint_dir=config["checkpoint_dir"]
    )

    # Visualize results
    visualize_sdf_slice(trained_model, device=config["device"])

if __name__ == "__main__":
    main()