import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from torchvision.utils import make_grid

try:
    import lpips
except ImportError:
    raise ImportError("Please install the lpips package: pip install lpips")

def run_latent_space_analysis(model, data_loader, rank=1, save_dir=".research/iteration1/images"):
    """
    Given a trained model and a data_loader, this function computes the latent
    representations, calculates pairwise Euclidean distances and perceptual (LPIPS)
    distances, computes the Pearson correlation coefficient, and saves a t-SNE plot.
    """
    print("Starting Latent Space Analysis ...")
    model.eval()
    latent_vectors_list = []
    lpips_dists_list = []

    loss_fn = lpips.LPIPS(net='alex').to(next(model.parameters()).device)
    
    with torch.no_grad():
        for batch in data_loader:
            real_data = batch[0].to(next(model.parameters()).device)
            latents = model.netQ(real_data, rank)
            latent_vectors_list.append(latents.cpu().numpy())
            
            gen_images = model.netG(latents)
            real_scaled = real_data * 2 - 1
            gen_scaled = gen_images * 2 - 1
            d = loss_fn(real_scaled, gen_scaled)
            lpips_dists_list.append(d.view(-1).cpu().numpy())
    
    latent_vectors = np.concatenate(latent_vectors_list, axis=0)
    lpips_dists = np.concatenate(lpips_dists_list, axis=0)
    
    sample_size = min(len(lpips_dists), latent_vectors.shape[0])
    sample_latents = latent_vectors[:sample_size]
    latent_pairwise_distance = pdist(sample_latents, metric='euclidean')
    
    num_pairs = min(len(lpips_dists), latent_pairwise_distance.shape[0])
    lpips_sample = lpips_dists[:num_pairs]
    latent_pairwise_distance = latent_pairwise_distance[:num_pairs]
    
    corr_coef, p_value = pearsonr(latent_pairwise_distance, lpips_sample)
    print(f"Pearson correlation coefficient between latent Euclidean and LPIPS distances: {corr_coef:.4f} (p={p_value:.4g})")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, latent_vectors.shape[0]-1))
    latent_tsne = tsne.fit_transform(latent_vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c='blue', alpha=0.6)
    plt.title("t-SNE Visualization of IL-WGAN Latent Space")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig(f"{save_dir}/latent_tsne_visualization.pdf", bbox_inches="tight")
    plt.close()
    print(f"Latent space t-SNE plot saved ({save_dir}/latent_tsne_visualization.pdf).")

def create_latent_interpolation_plot(model, real_data, rank, save_dir=".research/iteration1/images"):
    """Create and save latent interpolation visualization."""
    model.eval()
    with torch.no_grad():
        latent_a = model.netQ(real_data, rank)
        latent_b = model.netQ(real_data.flip(0), rank)
        num_interp = 8
        interp_latents = []
        for alpha in np.linspace(0, 1, num_interp):
            interp = latent_a * (1-alpha) + latent_b * alpha
            interp_latents.append(interp)
        
        interp_images = [model.netG(z) for z in interp_latents]
        interp_images = torch.stack([imgs[0] for imgs in interp_images])
        grid = make_grid(interp_images, nrow=num_interp, normalize=True)
        plt.figure(figsize=(12, 2))
        plt.imshow(np.transpose(grid.cpu().detach().numpy(), (1,2,0)))
        plt.title("Latent Interpolation (IL-WGAN)")
        plt.axis('off')
        plt.savefig(f"{save_dir}/latent_interpolation_IL-WGAN.pdf", bbox_inches="tight")
        plt.close()

def plot_adaptive_lambda_evolution(epochs, lambda_values, save_dir=".research/iteration1/images"):
    """Plot the evolution of lambda_iso over epochs."""
    plt.figure()
    plt.plot(epochs, lambda_values, marker='o')
    plt.title("Adaptive Weight of Isometric Loss")
    plt.xlabel("Epoch")
    plt.ylabel("lambda_iso")
    plt.grid(True)
    plt.savefig(f"{save_dir}/adaptive_lambda_iso.pdf", bbox_inches="tight")
    plt.close()
    print(f"Adaptive weighting experiment plot saved ({save_dir}/adaptive_lambda_iso.pdf).")
