import torch
import torch.utils.data
import numpy as np

def get_dummy_dataloader(batch_size=16, num_batches=5):
    """
    Create a dummy data loader with random images for testing.
    Returns a DataLoader with random images in [0,1] range.
    """
    dummy_data = torch.rand(batch_size * num_batches, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (batch_size * num_batches,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader

def create_dummy_dataset(num_samples=64, image_size=(3, 32, 32)):
    """
    Create a dummy dataset of random images for experiments.
    Returns tensor of shape (num_samples, channels, height, width).
    """
    return torch.rand(num_samples, *image_size)

def gen_noise_with_rank(batch_size, z_dim, rank, device):
    """
    Generate random noise for the generator.
    In this dummy implementation, rank is not used but kept for compatibility.
    """
    noise = torch.randn(batch_size, z_dim, device=device)
    return noise
