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
# Set up results directory for Experiment 3
# -------------------------
RESULTS_DIR = os.path.join(os.getcwd(), "experiment_3")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Embedding class definitions
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
        self.register_buffer('pos_encoding', self._get_positional_encoding(max_len, hidden_size), persistent=False)

    def _get_positional_encoding(self, max_len, hidden_size):
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        x = self.linear(x)
        L = x.size(1)
        x = x + self.pos_encoding[:, :L, :]
        return x

class SparseAutoencoderEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, sparsity_weight=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_dim)
        self.activation = nn.ReLU()
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        B, L, input_dim = x.shape
        x_flat = x.reshape(B * L, input_dim)
        encoded = self.activation(self.encoder(x_flat))
        decoded = self.decoder(encoded)
        encoded = encoded.reshape(B, L, -1)
        decoded = decoded.reshape(B, L, input_dim)
        return encoded

    def get_embedding(self, x):
        return self.forward(x)

    def loss(self, x, decoded, encoded):
        recon_loss = torch.mean((x - decoded) ** 2)
        sparsity_loss = torch.mean(torch.abs(encoded))
        return recon_loss + self.sparsity_weight * sparsity_loss

class SpectralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        nn.init.orthogonal_(self.linear.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# -------------------------
# Settings and Data Loading
# -------------------------
# Adjust path as necessary
F_path = os.path.join("data_prepped_for_models", "neural_data_matched.npy")
Farray = np.load(F_path)  # Farray shape: (num_neurons, num_timesteps)
num_neurons, num_timesteps = Farray.shape
print("Neural data shape:", Farray.shape)
hidden_size = 4096

# -------------------------
# Prepare tokens
# -------------------------
# Each token is a column of Farray. Create a tensor of shape (num_timesteps, 1, num_neurons)
tokens_tensor = torch.tensor(Farray.T, dtype=torch.float32).unsqueeze(1)

# -------------------------
# Instantiate embedding modules
# -------------------------
embeddings = {
    "vanilla": VanillaLinearEmbedding(num_neurons, hidden_size),
    "positional": PositionalLinearEmbedding(num_neurons, hidden_size),
    "spectral": SpectralEmbedding(num_neurons, hidden_size),
    "sparse": SparseAutoencoderEmbedding(num_neurons, hidden_size)
}
embedding_titles = [
    "Linear",
    "Positional Encoding",
    "Laplacian Eigenmap",
    "Sparse Autoencoder"
]
embedding_order = ["vanilla", "positional", "spectral", "sparse"]

# -------------------------
# Define a plotting function
# -------------------------
def plot_embeddings(norm_activity, cbar_label, filename):
    """
    Plot PCA results of embeddings, coloring tokens by norm_activity.
    norm_activity: array of shape (num_tokens,) with values in [0,1]
    cbar_label: string for the colorbar label.
    filename: file name for saving the plot.
    """
    fig = plt.figure(figsize=(30, 6), dpi=300)
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.08], wspace=0.7)
    axes = [fig.add_subplot(gs[i], projection='3d') for i in range(4)]
    cbar_ax = fig.add_axes([0.9, 0.375, 0.02, 0.25])
    
    last_scatter = None
    for i, emb_key in enumerate(embedding_order):
        module = embeddings[emb_key]
        module.eval()
        with torch.no_grad():
            emb_out = module(tokens_tensor)
        emb_out = emb_out.squeeze(1).cpu().numpy()  # shape: (num_tokens, hidden_size)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(emb_out)
    
        ax = axes[i]
        ax.set_title(embedding_titles[i], fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    
        sc = ax.scatter(
            pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
            c=norm_activity, cmap='bwr', s=30, alpha=0.8
        )
        last_scatter = sc
    
        max_range = np.array([
            pca_result[:, 0].max() - pca_result[:, 0].min(),
            pca_result[:, 1].max() - pca_result[:, 1].min(),
            pca_result[:, 2].max() - pca_result[:, 2].min()
        ]).max() / 2.0
        mid_x = (pca_result[:, 0].max() + pca_result[:, 0].min()) * 0.5
        mid_y = (pca_result[:, 1].max() + pca_result[:, 1].min()) * 0.5
        mid_z = (pca_result[:, 2].max() + pca_result[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label(cbar_label)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

# -------------------------
# Plot 1: Color tokens by normalized sum of all absolute activity
# -------------------------
activity_all = np.sum(np.abs(Farray), axis=0)
act_min_all, act_max_all = activity_all.min(), activity_all.max()
norm_activity_all = (activity_all - act_min_all) / (act_max_all - act_min_all)
print("Plot 1 using sum of all activity.")
plot_embeddings(norm_activity_all, "Norm. Abs. Pop. Activity", "pca_all_activity.png")

# -------------------------
# Plot 2: Color tokens by normalized sum of top 50 absolute neuron responses
# -------------------------
n_tokens = num_timesteps
activity_top50 = np.array([
    np.sum(np.sort(np.abs(Farray[:, i]))[-50:]) for i in range(n_tokens)
])
act_min_top50, act_max_top50 = activity_top50.min(), activity_top50.max()
norm_activity_top50 = (activity_top50 - act_min_top50) / (act_max_top50 - act_min_top50)
print("Plot 2 using sum of top 50 neuron responses.")
plot_embeddings(norm_activity_top50, "Norm. Abs. Top Responders", "pca_top50_activity.png")
