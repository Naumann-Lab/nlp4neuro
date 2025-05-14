#!/usr/bin/env python3
import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import matplotlib.gridspec as gridspec

# -------------------------
# Configuration
# -------------------------
fish_num = 9
DATA_DIR   = "/hpc/group/naumannlab/jjm132/data_prepped_for_models"
SAL_DIR    = f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/deepseek_only/fish{fish_num}"
RESULTS_DIR = os.path.join(
    "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3b_results",
    "experiment_3_pca"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

hidden_size    = 4096
n_remove       = 50
random_seed    = 0

# -------------------------
# Embedding modules (from exp3b)
# -------------------------
class VanillaLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__(); self.linear = nn.Linear(input_dim, hidden_size)
    def forward(self, x): return self.linear(x)

class PositionalLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_len=1024):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.register_buffer('pos_encoding', self._make_pe(max_len, hidden_size), persistent=False)
    def _make_pe(self, max_len, d):
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:,0::2], pe[:,1::2] = torch.sin(pos*div), torch.cos(pos*div)
        return pe.unsqueeze(0)
    def forward(self, x):
        z = self.linear(x)
        L = z.size(1)
        return z + self.pos_encoding[:,:L]

class RelPosLinearEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, max_dist=32):
        super().__init__()
        self.linear   = nn.Linear(input_dim, hidden_size)
        self.max_dist = max_dist
        self.rel      = nn.Embedding(2*max_dist+1, hidden_size)
    def forward(self, x):
        z = self.linear(x)
        B,L,_ = z.shape
        offs = torch.arange(L, device=z.device) - torch.arange(L, device=z.device)[0]
        offs = torch.clamp(offs, -self.max_dist, self.max_dist) + self.max_dist
        return z + self.rel(offs).unsqueeze(0).expand(B,L,-1)

class SpectralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.linear     = nn.Linear(input_dim, hidden_size)
        nn.init.orthogonal_(self.linear.weight)
        self.activation = nn.Tanh()
    def forward(self, x): return self.activation(self.linear(x))

class SparseAutoencoderEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_size, sparsity_weight=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
    def forward(self, x):
        B,L,D = x.shape
        flat = x.view(B*L, D)
        z    = self.activation(self.encoder(flat))
        return z.view(B,L,-1)

def get_embeddings(input_dim):
    return {
        "vanilla":    VanillaLinearEmbedding(input_dim, hidden_size),
        "positional": PositionalLinearEmbedding(input_dim, hidden_size),
        "relativepos":RelPosLinearEmbedding(input_dim, hidden_size),
        "spectral":   SpectralEmbedding(input_dim, hidden_size),
        "sparse":     SparseAutoencoderEmbedding(input_dim, hidden_size),
    }

embedding_order  = ["vanilla", "positional", "relativepos", "spectral", "sparse"]
embedding_titles = [
    "Linear",
    "Positional Encoding",
    "Relative Position",
    "Laplacian Eigenmap",
    "Sparse Autoencoder",
]

# -------------------------
# 3D‐PCA (via TruncatedSVD) plotting
# -------------------------
def plot_embeddings(tokens, embeddings, fname):
    fig = plt.figure(figsize=(30,6), dpi=300)
    gs  = gridspec.GridSpec(1, 6, width_ratios=[1]*5+[0.08], wspace=0.7)
    axes = [fig.add_subplot(gs[i], projection='3d') for i in range(5)]
    cax  = fig.add_axes([0.92,0.375,0.02,0.25])

    scatter = None
    for i,key in enumerate(embedding_order):
        module = embeddings[key].eval()
        with torch.no_grad():
            emb_out = module(tokens).squeeze(1).cpu().numpy()
        svd = TruncatedSVD(n_components=3, random_state=0)
        coords = svd.fit_transform(emb_out)

        ax = axes[i]
        ax.set_title(embedding_titles[i], fontsize=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

        scatter = ax.scatter(
            coords[:,0], coords[:,1], coords[:,2],
            s=20, alpha=0.6, marker='o'
        )

        # equal‐axis scaling
        mr = np.ptp(coords, axis=0).max()/2.0
        mid = coords.min(axis=0) + np.ptp(coords, axis=0)/2.0
        ax.set_xlim(mid[0]-mr, mid[0]+mr)
        ax.set_ylim(mid[1]-mr, mid[1]+mr)
        ax.set_zlim(mid[2]-mr, mid[2]+mr)

    cbar = fig.colorbar(scatter, cax=cax, orientation='vertical')
    cbar.remove()   # no color - single hue
    fig.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95)
    outpath = os.path.join(RESULTS_DIR, fname)
    plt.savefig(outpath, transparent=True)
    print("Saved:", outpath)
    plt.clf()

# -------------------------
# Load full data & saliency
# -------------------------
F = np.load(f"{DATA_DIR}/fish{fish_num}_neural_data_matched.npy")
sal = np.load(f"{SAL_DIR}/fish{fish_num}_importance.npy")
salient50 = sal.argsort()[-n_remove:][::-1]

# -------------------------
# Two removal scenarios
# -------------------------
np.random.seed(random_seed)
rand50 = np.random.choice(F.shape[0], n_remove, replace=False)

for scenario, idx in [("random50_removed", rand50),
                      ("salient50_removed", salient50)]:
    # remove neurons, build tokens & embeddings
    reduced = np.delete(F, idx, axis=0)
    tokens  = torch.tensor(reduced.T, dtype=torch.float32).unsqueeze(1)
    emb     = get_embeddings(reduced.shape[0])

    print(f"→ {scenario}: neuron count {reduced.shape[0]}")
    plot_embeddings(tokens, emb, f"pca_{scenario}.pdf")

print("✓ All done.")
