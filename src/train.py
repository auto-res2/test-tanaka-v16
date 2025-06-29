import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

try:
    import lpips
except ImportError:
    raise ImportError("Please install the lpips package: pip install lpips")

class DummyNetQ(nn.Module):
    """A dummy encoder: image -> latent vector."""
    def __init__(self, input_dim=32*32*3, latent_dim=64):
        super(DummyNetQ, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x, rank=0):
        x = x.view(x.size(0), -1)
        return torch.tanh(self.fc(x))

class DummyNetG(nn.Module):
    """A dummy generator: latent vector -> image."""
    def __init__(self, latent_dim=64, output_dim=32*32*3):
        super(DummyNetG, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
    
    def forward(self, z):
        out = torch.sigmoid(self.fc(z))
        return out.view(z.size(0), 3, 32, 32)

class DummyNetD(nn.Module):
    """A dummy discriminator: image -> scalar score."""
    def __init__(self, input_dim=32*32*3):
        super(DummyNetD, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x, rank=0):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LWGAN(nn.Module):
    def __init__(self, z_dim, netQ, netG, netD, device=torch.device("cpu")):
        super(LWGAN, self).__init__()
        self.z_dim = z_dim
        self.netQ = netQ
        self.netG = netG
        self.netD = netD
        self.device = device

    def D_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        post_data = self.netG(self.netQ(real_data, rank))
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = -torch.abs(diff) if abs else -diff
        return losses.mean()

    def GQ_loss(self, real_data, fake_data, rank: int, abs: bool = False):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        diff = self.netD(post_data, rank) - self.netD(fake_data, rank)
        losses = l2 + torch.abs(diff) if abs else l2 + diff
        return losses.mean()

    def gradient_penalty_D(self, x, z, rank: int):
        x_hat = self.netG(z)
        x_tilde = self.netG(self.netQ(x, rank))
        return ((x_hat - x_tilde)**2).mean()

    def mmd_penalty(self, real_data, rank: int, lambdammd: float):
        n = real_data.shape[0]
        mmd = torch.tensor(0.0, device=real_data.device)
        if lambdammd != 0.0:
            mmd = torch.mean(self.netQ(real_data, rank))
            mmd = lambdammd * mmd
        return mmd

    def recon_loss(self, real_data, rank: int):
        n = real_data.shape[0]
        post_data = self.netG(self.netQ(real_data, rank))
        l2 = torch.linalg.norm((real_data - post_data).view(n, -1), dim=-1)
        return l2.mean()

    def forward(self, x1, x2, rank: int, lambda_mmd: float, lambda_rank: float):
        n = x1.shape[0]
        noise = torch.randn(n, self.z_dim, device=self.device)
        fake_data = self.netG(noise)
        cost_GQ = self.GQ_loss(x1, fake_data, rank, abs=False)
        mmd = self.mmd_penalty(x2, rank, lambda_mmd)
        primal_cost = cost_GQ + mmd + lambda_rank * rank
        return primal_cost

def compute_isometric_loss(real_images, gen_images):
    """
    Compute isometric loss using lpips as a perceptual metric.
    """
    loss_fn = lpips.LPIPS(net='alex').to(real_images.device)
    real_images_scaled = real_images * 2 - 1
    gen_images_scaled = gen_images * 2 - 1
    distances = loss_fn(real_images_scaled, gen_images_scaled)
    return distances.mean()

def compute_total_loss(model, real_data, rank, fake_data, lambda_mmd, lambda_rank, lambda_iso):
    """
    Computes the total loss including the isometric loss (if lambda_iso > 0).
    """
    cost = model.GQ_loss(real_data, fake_data, rank, abs=False)
    cost += model.mmd_penalty(real_data, rank, lambda_mmd)
    cost += lambda_rank * rank
    
    latent_vectors = model.netQ(real_data, rank)
    images_generated = model.netG(latent_vectors)
    iso_loss = compute_isometric_loss(real_data, images_generated)
    total_loss = cost + lambda_iso * iso_loss
    return total_loss

def create_models(device):
    """Create and return the neural network models."""
    netQ = DummyNetQ().to(device)
    netG = DummyNetG().to(device)
    netD = DummyNetD().to(device)
    return netQ, netG, netD
