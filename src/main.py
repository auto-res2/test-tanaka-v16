#!/usr/bin/env python3
"""
IL-WGAN Experiment Code

This script implements:
  1. An ablation study of the isometric regularizer comparing IL-WGAN and LWGAN.
  2. An adaptive weighting experiment for the isometric loss.
  3. A visualization & quantitative analysis of latent space geometry.

The script uses PyTorch and auxiliary libraries:
  - torch, torchvision
  - numpy, matplotlib, seaborn
  - sklearn, scipy, lpips

Plots are saved in PDF format to .research/iteration1/images/ directory.
A quick test function "test_run" is provided at the end.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

from preprocess import get_dummy_dataloader, create_dummy_dataset, gen_noise_with_rank
from train import LWGAN, create_models, compute_total_loss
from evaluate import run_latent_space_analysis, create_latent_interpolation_plot, plot_adaptive_lambda_evolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_ablation_experiment(num_epochs=3, batch_size=16, lambda_mmd=0.1, lambda_rank=0.1, lambda_iso_IL=1.0, lambda_iso_base=0.0):
    """
    Run two training loops (IL-WGAN with isometric loss and LWGAN without)
    on a dummy dataset. For each, we print out the loss values.
    """
    print("Starting Ablation Experiment ...")
    
    netQ_il, netG_il, netD_il = create_models(device)
    netQ_base, netG_base, netD_base = create_models(device)
    model_il = LWGAN(z_dim=64, netQ=netQ_il, netG=netG_il, netD=netD_il, device=device)
    model_base = LWGAN(z_dim=64, netQ=netQ_base, netG=netG_base, netD=netD_base, device=device)
    optimizer_il = optim.Adam(model_il.parameters(), lr=0.001)
    optimizer_base = optim.Adam(model_base.parameters(), lr=0.001)
    
    dummy_images = create_dummy_dataset(64).to(device)
    
    for epoch in range(num_epochs):
        model_il.train()
        model_base.train()
        real_data = dummy_images[:batch_size]
        rank = 1
        noise_il = gen_noise_with_rank(batch_size, 64, rank, device)
        fake_data_il = netG_il(noise_il)
        
        optimizer_il.zero_grad()
        loss_il = compute_total_loss(model_il, real_data, rank, fake_data_il, lambda_mmd, lambda_rank, lambda_iso_IL)
        loss_il.backward()
        optimizer_il.step()
        
        noise_base = gen_noise_with_rank(batch_size, 64, rank, device)
        fake_data_base = netG_base(noise_base)
        optimizer_base.zero_grad()
        loss_base = compute_total_loss(model_base, real_data, rank, fake_data_base, lambda_mmd, lambda_rank, lambda_iso_base)
        loss_base.backward()
        optimizer_base.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | IL-WGAN Loss: {loss_il.item():.4f} | LWGAN Loss: {loss_base.item():.4f}")
    
    create_latent_interpolation_plot(model_il, real_data, rank)
    print("Ablation experiment completed with latent interpolation plot saved.")

def run_adaptive_weight_experiment(num_epochs=20, batch_size=16, lambda_mmd=0.1, lambda_rank=0.1, init_lambda_iso=1.0, adapt_interval=5):
    """
    Run training with adaptive weighting of the isometric loss.
    Every adapt_interval epochs, a dummy validation check is done and lambda_iso is adjusted.
    """
    print("Starting Adaptive Weighting Experiment ...")
    
    netQ, netG, netD = create_models(device)
    model = LWGAN(z_dim=64, netQ=netQ, netG=netG, netD=netD, device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    lambda_iso = init_lambda_iso
    dummy_val_losses = []
    dummy_images = create_dummy_dataset(64).to(device)
    
    epochs = []
    lambda_values = []
    
    for epoch in range(num_epochs):
        model.train()
        real_data = dummy_images[:batch_size]
        rank = 1
        noise = gen_noise_with_rank(batch_size, 64, rank, device)
        fake_data = netG(noise)
        optimizer.zero_grad()
        loss = compute_total_loss(model, real_data, rank, fake_data, lambda_mmd, lambda_rank, lambda_iso)
        loss.backward()
        optimizer.step()
        
        val_loss = loss.item() + np.random.randn()*0.01
        dummy_val_losses.append(val_loss)
        epochs.append(epoch + 1)
        lambda_values.append(lambda_iso)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | lambda_iso: {lambda_iso:.4f} | Val Loss: {val_loss:.4f}")
        
        if (epoch+1) % adapt_interval == 0 and len(dummy_val_losses) >= 2:
            if dummy_val_losses[-1] > dummy_val_losses[-2]:
                lambda_iso *= 0.9
                print(f"Adaptive update: Decreased lambda_iso to {lambda_iso:.4f}")
            else:
                lambda_iso *= 1.1
                print(f"Adaptive update: Increased lambda_iso to {lambda_iso:.4f}")
    
    plot_adaptive_lambda_evolution(epochs, lambda_values)
    print("Adaptive weighting experiment completed.")

def run_latent_space_geometry_experiment():
    """
    Run the latent space analysis experiment.
    """
    print("Starting Latent Space Geometry Experiment ...")
    
    netQ, netG, netD = create_models(device)
    model = LWGAN(z_dim=64, netQ=netQ, netG=netG, netD=netD, device=device)
    data_loader = get_dummy_dataloader(batch_size=8, num_batches=3)
    
    run_latent_space_analysis(model, data_loader, rank=1)
    print("Latent space geometry experiment completed.")

def test_run():
    """
    Run a quick version of all experiments to verify functionality.
    """
    print("----- Running Quick Test of All Experiments -----")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    print("\n=== Experiment 1: Ablation Study ===")
    run_ablation_experiment(num_epochs=2, batch_size=8, lambda_mmd=0.05, lambda_rank=0.05, lambda_iso_IL=1.0, lambda_iso_base=0.0)
    
    print("\n=== Experiment 2: Adaptive Weighting ===")
    run_adaptive_weight_experiment(num_epochs=10, batch_size=8, lambda_mmd=0.05, lambda_rank=0.05, init_lambda_iso=1.0, adapt_interval=3)
    
    print("\n=== Experiment 3: Latent Space Analysis ===")
    run_latent_space_geometry_experiment()
    
    print("\n----- All Tests Completed Successfully -----")

def main():
    """
    Main execution function that runs all three experiments.
    """
    print("=" * 80)
    print("IL-WGAN (Isometric Latent Wasserstein GAN) Experimental Framework")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 80)
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: ABLATION STUDY")
    print("="*50)
    print("Comparing IL-WGAN (with isometric regularization) vs LWGAN (baseline)")
    run_ablation_experiment(num_epochs=5, batch_size=16, lambda_mmd=0.1, lambda_rank=0.1, lambda_iso_IL=1.0, lambda_iso_base=0.0)
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: ADAPTIVE WEIGHTING")
    print("="*50)
    print("Testing adaptive weighting mechanism for isometric loss")
    run_adaptive_weight_experiment(num_epochs=20, batch_size=16, lambda_mmd=0.1, lambda_rank=0.1, init_lambda_iso=1.0, adapt_interval=5)
    
    print("\n" + "="*50)
    print("EXPERIMENT 3: LATENT SPACE GEOMETRY ANALYSIS")
    print("="*50)
    print("Analyzing latent space structure and geometric properties")
    run_latent_space_geometry_experiment()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("="*80)
    print("Generated PDF plots saved in: .research/iteration1/images/")
    print("- latent_interpolation_IL-WGAN.pdf")
    print("- adaptive_lambda_iso.pdf") 
    print("- latent_tsne_visualization.pdf")
    print("="*80)
    
    status_enum = "stopped"
    print(f"Status: {status_enum}")

if __name__ == "__main__":
    test_run()
