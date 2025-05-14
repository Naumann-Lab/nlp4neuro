#!/usr/bin/env python3

# idea -> salient neurons drive population responses... when removed the distance in pc space should decrease.
# if I remove a random neuron, it should not change the system much... if remove important, should change system.
"""Gradient‑coloured PCA centroids vs. *how many* salient neurons were removed
------------------------------------------------------------------------------
For k = 0…50 we drop **k salient + (50−k) random** neurons, build embeddings, and
plot **one centroid point per k** in 3‑D PC space (common basis per embedding).

Colour: k/50 → blue (0 salient) → red (50 salient)
Outputs
-------
• `pca_centroids_gradient.pdf` 5 panels (Linear…SparseAE) with 51 coloured points
• `centroid_coords.csv` embedding,k,pc1,pc2,pc3  (easy for table).
"""
import os, math, csv, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from sklearn.decomposition import TruncatedSVD

fish_num, hidden_size, n_drop = 9, 4096, 50
np.random.seed(0)
DATA_DIR = "/hpc/group/naumannlab/jjm132/data_prepped_for_models"
SAL_DIR  = f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/deepseek_only/fish{fish_num}"
OUT_DIR  = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3b_results/experiment_3_pca"
os.makedirs(OUT_DIR, exist_ok=True)

# ── embeddings ────────────────────────────────────────────────────────────
class Vanilla(nn.Module):
    def __init__(self,D,H): super().__init__(); self.l=nn.Linear(D,H)
    def forward(self,x): return self.l(x)
class PosEnc(nn.Module):
    def __init__(self,D,H,L=1024):
        super().__init__(); self.l=nn.Linear(D,H)
        pe=torch.zeros(L,H);pos=torch.arange(L).unsqueeze(1);
        div=torch.exp(torch.arange(0,H,2)*(-math.log(10000.0)/H))
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*div),torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0),persistent=False)
    def forward(self,x): z=self.l(x); return z+self.pe[:,:z.size(1)]
class RelPos(nn.Module):
    def __init__(self,D,H,R=32):
        super().__init__(); self.l=nn.Linear(D,H); self.R=R; self.rel=nn.Embedding(2*R+1,H)
    def forward(self,x):
        z=self.l(x);B,L,_=z.shape ; offs=torch.arange(L,device=z.device)-torch.arange(L,device=z.device)[0]
        offs=torch.clamp(offs,-self.R,self.R)+self.R
        return z+self.rel(offs).unsqueeze(0).expand(B,L,-1)
class Spectral(nn.Module):
    def __init__(self,D,H): super().__init__(); self.l=nn.Linear(D,H); nn.init.orthogonal_(self.l.weight)
    def forward(self,x): return torch.tanh(self.l(x))
class SparseAE(nn.Module):
    def __init__(self,D,H): super().__init__(); self.enc=nn.Linear(D,H)
    def forward(self,x): B,L,_=x.shape; return torch.relu(self.enc(x.view(B*L,-1))).view(B,L,-1)

def embeds(D):
    return {"Linear":Vanilla(D,hidden_size),"PosEnc":PosEnc(D,hidden_size),"RelPos":RelPos(D,hidden_size),"Spectral":Spectral(D,hidden_size),"SparseAE":SparseAE(D,hidden_size)}
order=["Linear","PosEnc","RelPos","Spectral","SparseAE"]

# ── data & saliency indices ───────────────────────────────────────────────
F   = np.load(f"{DATA_DIR}/fish{fish_num}_neural_data_matched.npy")
SAL = np.load(f"{SAL_DIR}/fish{fish_num}_importance.npy")
sal_idx_sorted = SAL.argsort()[::-1]  # highest first

all_indices = np.arange(F.shape[0])
Tok = lambda idx: torch.tensor(F[idx].T, dtype=torch.float32).unsqueeze(1)

# ── storage for CSV ───────────────────────────────────────────────────────
csv_rows = [("embedding","k_salient","pc1","pc2","pc3")]

# ── figure setup ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(30,6), dpi=300)
gs  = gridspec.GridSpec(1, len(order)+1, width_ratios=[1]*len(order)+[0.05], wspace=0.6)
axes=[fig.add_subplot(gs[i],projection='3d') for i in range(len(order))]
cax = fig.add_axes([0.92,0.35,0.02,0.3])

for emb_name,ax in zip(order,axes):
    print(f"Embedding {emb_name}…", flush=True)
    # baseline count after dropping 50 neurons
    keep_0 = np.setdiff1d(all_indices, np.random.choice(all_indices, n_drop, replace=False))
    D_rem  = keep_0.size
    E = embeds(D_rem)[emb_name].eval()
    # basis fit once on k=0 (all random) tokens
    with torch.no_grad():
        base_tokens = Tok(keep_0)
        Z_base = E(base_tokens).squeeze(1).cpu().numpy()
    svd = TruncatedSVD(n_components=3, random_state=0).fit(Z_base)

    for k in range(n_drop+1):  # k salient removed
        sal_to_drop = sal_idx_sorted[:k]
        other_pool  = np.setdiff1d(all_indices, sal_to_drop)
        rand_to_drop= np.random.choice(other_pool, n_drop-k, replace=False)
        drop_idx    = np.concatenate([sal_to_drop, rand_to_drop])
        keep_idx    = np.setdiff1d(all_indices, drop_idx)
        assert keep_idx.size == D_rem

        with torch.no_grad():
            Z = svd.transform(E(Tok(keep_idx)).squeeze(1).cpu().numpy())
        centroid = Z.mean(0)
        csv_rows.append((emb_name,k,*centroid))

        col = plt.cm.bwr(k/n_drop)
        ax.scatter(*centroid, c=[col], s=60, edgecolor='k', linewidth=0.3)
        if k in (0,25,50):
            ax.text(*(centroid+0.02), str(k), fontsize=8, color='k')

    ax.set_title(emb_name)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    # equal axes
    pts=np.array([r[2:] for r in csv_rows if r[0]==emb_name])
    mr=np.ptp(pts,0).max()/2.0 ; mid=pts.min(0)+np.ptp(pts,0)/2.0
    for d,l in zip('xyz',mid): getattr(ax,f'set_{d}lim')(l-mr,l+mr)

# colourbar
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=n_drop)
cbar = mpl.colorbar.ColorbarBase(cax, cmap='bwr', norm=norm)
cbar.set_label('# salient neurons removed')

fig.subplots_adjust(left=0.04,right=0.9,top=0.95,bottom=0.05)
fig.savefig(os.path.join(OUT_DIR,'pca_centroids_gradient.pdf'), transparent=True)
print('Saved pca_centroids_gradient.pdf')

# CSV
with open(os.path.join(OUT_DIR,'centroid_coords.csv'),'w',newline='') as f:
    writer=csv.writer(f); writer.writerows(csv_rows)
print('Centroid coordinates written to centroid_coords.csv')
