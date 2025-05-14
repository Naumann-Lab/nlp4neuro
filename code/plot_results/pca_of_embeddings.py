#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

# -------------------------
# Set up results directory for Experiment 3 PCA
# -------------------------
fish_num = 9

RESULTS_DIR = os.path.join("/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3b_results", "experiment_3_pca")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = "/hpc/group/naumannlab/jjm132/data_prepped_for_models"
DATA_DIR2 = f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/deepseek_only/fish{fish_num}"

# -------------------------
# Embedding class definitions (matching experiment 3b)
# -------------------------
class VanillaLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)

    def forward(self, x):
        return self.linear(x)

class PositionalLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_len=1024):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.register_buffer(
            'pos_encoding',
            self._get_positional_encoding(max_len, hidden_size),
            persistent=False
        )

    def _get_positional_encoding(self, max_len, hidden_size):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.linear(x)
        L = x.size(1)
        x = x + self.pos_encoding[:, :L, :]
        return x

class RelPosLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_dist=32):
        super().__init__()
        self.linear    = nn.Linear(input_dim, hidden_size)
        self.max_dist  = max_dist
        self.rel       = nn.Embedding(2 * max_dist + 1, hidden_size)

    def forward(self, x):
        z = self.linear(x)
        B, L, _ = z.shape
        offs = torch.arange(L, device=z.device) - torch.arange(L, device=z.device)[0]
        offs = torch.clamp(offs, -self.max_dist, self.max_dist) + self.max_dist
        return z + self.rel(offs).unsqueeze(0).expand(B, L, -1)

class SpectralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear     = nn.Linear(input_dim, hidden_size)
        nn.init.orthogonal_(self.linear.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))

class SparseAutoencoderEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, sparsity_weight=1e-3):
        super().__init__()
        self.encoder         = nn.Linear(input_dim, hidden_size)
        self.decoder         = nn.Linear(hidden_size, input_dim)
        self.activation      = nn.ReLU()
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        B, L, input_dim = x.shape
        x_flat = x.reshape(B * L, input_dim)
        encoded = self.activation(self.encoder(x_flat))
        decoded = self.decoder(encoded)
        encoded = encoded.reshape(B, L, -1)
        decoded = decoded.reshape(B, L, input_dim)
        return encoded

# -------------------------
# Helper to instantiate embedding modules
# -------------------------
hidden_size = 4096

def get_embeddings(input_dim):
    return {
        "vanilla":    VanillaLinearEmbedding(input_dim, hidden_size),
        "positional": PositionalLinearEmbedding(input_dim, hidden_size),
        "relativepos":RelPosLinearEmbedding(input_dim, hidden_size),
        "spectral":   SpectralEmbedding(input_dim, hidden_size),
        "sparse":     SparseAutoencoderEmbedding(input_dim, hidden_size)
    }

embedding_order  = ["vanilla", "positional", "relativepos", "spectral", "sparse"]
embedding_titles = [
    "Linear",
    "Positional Encoding",
    "Relative Position",
    "Laplacian Eigenmap",
    "Sparse Autoencoder"
]

# -------------------------
# PCA and plotting function
# -------------------------
def plot_embeddings(tokens_tensor, embeddings, norm_activity, cbar_label, filename):
    fig = plt.figure(figsize=(30, 6), dpi=300)
    # one extra column for colorbar
    gs = gridspec.GridSpec(1, 6, width_ratios=[1,1,1,1,1,0.08], wspace=0.7)
    axes = [fig.add_subplot(gs[i], projection='3d') for i in range(5)]
    cbar_ax = fig.add_axes([0.92, 0.375, 0.02, 0.25])

    last_scatter = None
    for i, key in enumerate(embedding_order):
        module = embeddings[key]
        module.eval()
        with torch.no_grad():
            emb_out = module(tokens_tensor)
        # shape: (num_tokens, hidden_size)
        emb_out = emb_out.squeeze(1).cpu().numpy()
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(emb_out)

        ax = axes[i]
        ax.set_title(embedding_titles[i], fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        sc = ax.scatter(
            pca_result[:,0], pca_result[:,1], pca_result[:,2],
            c=norm_activity, cmap='bwr', s=30, alpha=0.8
        )
        last_scatter = sc

        # equal aspect
        max_range = np.array([
            pca_result[:,0].max() - pca_result[:,0].min(),
            pca_result[:,1].max() - pca_result[:,1].min(),
            pca_result[:,2].max() - pca_result[:,2].min()
        ]).max() / 2.0
        mid_x = (pca_result[:,0].max() + pca_result[:,0].min()) * 0.5
        mid_y = (pca_result[:,1].max() + pca_result[:,1].min()) * 0.5
        mid_z = (pca_result[:,2].max() + pca_result[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label(cbar_label)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

# -------------------------
# Load full neural data
# -------------------------
Farray = np.load(f"{DATA_DIR}/fish{fish_num}_neural_data_matched.npy")
num_neurons, num_timesteps = Farray.shape
print("Neural data shape:", Farray.shape)

# prepare tokens for full data
tokens_full = torch.tensor(Farray.T, dtype=torch.float32).unsqueeze(1)
emb_full    = get_embeddings(num_neurons)

# compute activity measures on full data
# (1) sum of abs activity
a1 = np.sum(np.abs(Farray), axis=0)
a1 = (a1 - a1.min()) / (a1.max() - a1.min())
# (2) sum of top50 absolute responses
a2 = np.array([np.sum(np.sort(np.abs(Farray[:,i]))[-50:]) for i in range(num_timesteps)])
a2 = (a2 - a2.min()) / (a2.max() - a2.min())

# Plot full-data embeddings
print("Plot: Full data, sum of all abs activity")
plot_embeddings(tokens_full, emb_full, a1, "Norm. Abs. Pop. Activity", "pca_full_all_activity.png")
print("Plot: Full data, sum of top50 abs activity")
plot_embeddings(tokens_full, emb_full, a2, "Norm. Abs. Top50 Response", "pca_full_top50_activity.png")

# -------------------------
# Remove top50 most-salient neurons and re-run PCA
# -------------------------
# load precomputed saliency values
sal = np.load(f"{DATA_DIR2}/fish9_importance.npy")  # array of saliency scores per neuron
salient50_idx = sal.argsort()[-50:][::-1]
# remove those neurons
dims = np.delete(Farray, salient50_idx, axis=0)
num_neurons_r, _ = dims.shape
print("After removal, neuron count:", num_neurons_r)

# prepare tokens and embeddings for reduced data
tokens_rem = torch.tensor(dims.T, dtype=torch.float32).unsqueeze(1)
emb_rem    = get_embeddings(num_neurons_r)

# recompute activity on reduced data
a1_r = np.sum(np.abs(dims), axis=0)
a1_r = (a1_r - a1_r.min()) / (a1_r.max() - a1_r.min())

def run_removal_plot():
    print("Plot: Removed top50 salient, sum of all abs activity")
    plot_embeddings(tokens_rem, emb_rem, a1_r,
                    "Norm. Abs. Pop. Activity (salient50 removed)",
                    "pca_removed_salient50_all_activity.png")

run_removal_plot()

print("✓ PCA analyses complete.")
