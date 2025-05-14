#!/usr/bin/env python3
"""PCA overlay *plus* KL‑divergence diagnostics
------------------------------------------------
Blue = random‑50 removed (0) • Red = salient‑50 removed (1)
For every embedding we now print **KL( RND || SAL )** and **KL( SAL || RND )**
using the 3‑D PC representation and a *diagonal‐covariance Gaussian* model.
"""
import os, math, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import TruncatedSVD

# ── paths / params ─────────────────────────────────────────────────────────
fish_num   = 9
DATA_DIR   = "/hpc/group/naumannlab/jjm132/data_prepped_for_models"
SAL_DIR    = f"/hpc/group/naumannlab/jjm132/nlp4neuro/results/deepseek_only/fish{fish_num}"
OUT_DIR    = "/hpc/group/naumannlab/jjm132/nlp4neuro/results/experiment3b_results/experiment_3_pca"
os.makedirs(OUT_DIR, exist_ok=True)

hidden_size = 4096; n_remove = 50; np.random.seed(0)

# ── embeddings (unchanged) ────────────────────────────────────────────────
class Vanilla(nn.Module):
    def __init__(self,D,H): super().__init__(); self.l=nn.Linear(D,H)
    def forward(self,x): return self.l(x)
class PosEnc(nn.Module):
    def __init__(self,D,H,L=1024):
        super().__init__(); self.l=nn.Linear(D,H)
        pe=torch.zeros(L,H);pos=torch.arange(L).unsqueeze(1)
        div=torch.exp(torch.arange(0,H,2)*(-math.log(10000.0)/H))
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*div),torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0),persistent=False)
    def forward(self,x): z=self.l(x); return z+self.pe[:,:z.size(1)]
class RelPos(nn.Module):
    def __init__(self,D,H,R=32):
        super().__init__(); self.l=nn.Linear(D,H); self.R=R; self.rel=nn.Embedding(2*R+1,H)
    def forward(self,x):
        z=self.l(x);B,L,_=z.shape
        offs=torch.arange(L,device=z.device)-torch.arange(L,device=z.device)[0]
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

# ── data subsets ─────────────────────────────────────────────────────────
F   = np.load(f"{DATA_DIR}/fish{fish_num}_neural_data_matched.npy")
SAL = np.load(f"{SAL_DIR}/fish{fish_num}_importance.npy")
rand_idx = np.random.choice(F.shape[0], n_remove, replace=False)
keep_rand = np.setdiff1d(np.arange(F.shape[0]), rand_idx)
keep_sal  = np.setdiff1d(np.arange(F.shape[0]), SAL.argsort()[-n_remove:][::-1])
Tok = lambda idx: torch.tensor(F[idx].T, dtype=torch.float32).unsqueeze(1)
Tok_base, Tok_R, Tok_S = Tok(keep_rand), Tok(keep_rand), Tok(keep_sal)

# ── helper: diag‑cov KL ───────────────────────────────────────────────────
def diag_KL(mu_p,mu_q,var_p,var_q):
    return 0.5*( np.sum(np.log(var_q/var_p)) - len(mu_p) + np.sum(var_p/var_q) + np.sum((mu_q-mu_p)**2/var_q) )

# ── plotting + KL computation ────────────────────────────────────────────
fig = plt.figure(figsize=(30,6), dpi=300)
gs  = gridspec.GridSpec(1, len(order)+1, width_ratios=[1]*len(order)+[0.06], wspace=0.6)
axes = [fig.add_subplot(gs[i], projection='3d') for i in range(len(order))]
cax  = fig.add_axes([0.92, 0.35, 0.02, 0.3])

kl_table = {}

for i,name in enumerate(order):
    E = embeds(keep_rand.size)[name].eval()
    with torch.no_grad():
        Z_base = E(Tok_base).squeeze(1).cpu().numpy()
        Z_R    = E(Tok_R   ).squeeze(1).cpu().numpy()
        Z_S    = E(Tok_S   ).squeeze(1).cpu().numpy()
    svd = TruncatedSVD(n_components=3, random_state=0).fit(Z_base)
    PR, PS = svd.transform(Z_R), svd.transform(Z_S)

    # KL in 3‑D PC space (diagonal cov)
    muR, muS = PR.mean(0), PS.mean(0)
    varR, varS = PR.var(0) + 1e-8, PS.var(0) + 1e-8  # eps for stability
    kl_RS = diag_KL(muR,muS,varR,varS)
    kl_SR = diag_KL(muS,muR,varS,varR)
    kl_table[name] = (kl_RS, kl_SR)

    # scatter coloured 0→1 (blue→red)
    pts   = np.vstack([PR,PS])
    lbl   = np.concatenate([np.zeros(PR.shape[0]), np.ones(PS.shape[0])])
    ax    = axes[i]
    sc    = ax.scatter(pts[:,0],pts[:,1],pts[:,2], c=lbl, cmap='bwr', vmin=0, vmax=1, s=6, alpha=0.6)
    ax.set_title(f"{name}\nKL R→S: {kl_RS:.2f}")
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    # equal limits
    mr=np.ptp(pts,0).max()/2.0; mid=pts.min(0)+np.ptp(pts,0)/2.0
    for d,l in zip('xyz',mid): getattr(ax,f'set_{d}lim')(l-mr,l+mr)

cbar=fig.colorbar(sc,cax=cax,ticks=[0,1]); cbar.ax.set_yticklabels(['Random-50','Salient-50'])
fig.subplots_adjust(left=0.04,right=0.9,top=0.95,bottom=0.05)
fig.savefig(os.path.join(OUT_DIR,'pca_salient_vs_random_colorbar_KL.pdf'), transparent=True)
print('Saved pca_salient_vs_random_colorbar_KL.pdf')

# write KL values table
with open(os.path.join(OUT_DIR,'kl_divergence_table.txt'),'w') as f:
    f.write('Embedding\tKL(RND||SAL)\tKL(SAL||RND)\n')
    for n in order:
        f.write(f"{n}\t{kl_table[n][0]:.4f}\t{kl_table[n][1]:.4f}\n")
print('KL divergence values written to kl_divergence_table.txt')
