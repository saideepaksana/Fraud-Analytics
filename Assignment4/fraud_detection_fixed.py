"""
Fraud Detection using BiGAN and AnoGAN
Fixed version — addresses:
  1. MinMaxScaler instead of StandardScaler (matches Tanh range [-1,1])
  2. Label smoothing (real=0.9) for GAN training stability
  3. AnoGAN N_ITER reduced from 500 → 100 (much faster inference)
  4. Fixed broken f-string in summary report
  5. Cleaner BiGAN loss labels (no variable reuse confusion)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler          # FIX 1
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report, f1_score
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── 1. Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('creditcard.csv')
print('Shape:', df.shape)
print(df['Class'].value_counts())

# FIX 1: MinMaxScaler → data lives in [-1, 1] matching Tanh output
features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(df[features].values.astype(np.float32))
y = df['Class'].values

INPUT_DIM = X.shape[1]
print(f'Input dim: {INPUT_DIM}, range: [{X.min():.2f}, {X.max():.2f}]')

X_normal = X[y == 0]
X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=SEED)
X_test = np.vstack([X_val, X[y == 1]])
y_test = np.hstack([np.zeros(len(X_val)), np.ones(y.sum())])

BATCH_SIZE = 256
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
print(f'Train batches: {len(train_loader)}')

# ── 2. AnoGAN Architecture ───────────────────────────────────────────────────
LATENT_DIM = 32

class AnoGAN_Generator(nn.Module):
    def __init__(self, ldim=LATENT_DIM, odim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ldim, 64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64, 128),  nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, odim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class AnoGAN_Discriminator(nn.Module):
    def __init__(self, idim=INPUT_DIM):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(idim, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 128),  nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64),   nn.LeakyReLU(0.2),
        )
        self.cls = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x): return self.cls(self.feat(x))
    def get_features(self, x): return self.feat(x)

anogan_G = AnoGAN_Generator().to(DEVICE)
anogan_D = AnoGAN_Discriminator().to(DEVICE)

# ── 3. AnoGAN Training ───────────────────────────────────────────────────────
EPOCHS = 100
LR = 0.0002
criterion = nn.BCELoss()
opt_G = optim.Adam(anogan_G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(anogan_D.parameters(), lr=LR, betas=(0.5, 0.999))
ag_g, ag_d = [], []

print('\nTraining AnoGAN...')
for epoch in range(EPOCHS):
    gl, dl = 0, 0
    for (xb,) in train_loader:
        xb = xb.to(DEVICE)
        bsz = xb.size(0)
        # FIX 2: Label smoothing — real=0.9 prevents D from becoming too confident
        rl = torch.full((bsz, 1), 0.9).to(DEVICE)
        fl = torch.zeros(bsz, 1).to(DEVICE)

        opt_D.zero_grad()
        z = torch.randn(bsz, LATENT_DIM).to(DEVICE)
        fake = anogan_G(z).detach()
        d_loss = criterion(anogan_D(xb), rl) + criterion(anogan_D(fake), fl)
        d_loss.backward(); opt_D.step()

        opt_G.zero_grad()
        z = torch.randn(bsz, LATENT_DIM).to(DEVICE)
        g_loss = criterion(anogan_D(anogan_G(z)), torch.ones(bsz,1).to(DEVICE))
        g_loss.backward(); opt_G.step()

        gl += g_loss.item(); dl += d_loss.item()
    n = len(train_loader)
    ag_g.append(gl/n); ag_d.append(dl/n)
    if (epoch+1) % 20 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} D={ag_d[-1]:.4f} G={ag_g[-1]:.4f}')

plt.figure(figsize=(8,4))
plt.plot(ag_g, label='G'); plt.plot(ag_d, label='D')
plt.title('AnoGAN Training Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout(); plt.savefig('anogan_training_loss.png', dpi=150); plt.show()

# ── 4. AnoGAN Inference ──────────────────────────────────────────────────────
# FIX 3: N_ITER 500 → 100. Tabular data converges fast; 500 was needlessly slow
LAMBDA, N_ITER, LR_INF = 0.1, 100, 0.01
anogan_G.eval(); anogan_D.eval()

def anogan_score(Xq, lam=LAMBDA, n_iter=N_ITER, lr=LR_INF, bsz=256):
    scores = []
    for s in range(0, len(Xq), bsz):
        x = torch.FloatTensor(Xq[s:s+bsz]).to(DEVICE)
        z = torch.randn(len(x), LATENT_DIM, requires_grad=True, device=DEVICE)
        oz = optim.Adam([z], lr=lr)
        for _ in range(n_iter):
            oz.zero_grad()
            xh = anogan_G(z)
            rec = torch.mean(torch.abs(x - xh), dim=1)
            ff = anogan_D.get_features(x).detach()
            fg = anogan_D.get_features(xh)
            feat = torch.mean((ff-fg)**2, dim=1)
            ((1-lam)*rec + lam*feat).mean().backward()
            oz.step()
        with torch.no_grad():
            xh = anogan_G(z)
            rec = torch.mean(torch.abs(x - xh), dim=1)
            feat = torch.mean((anogan_D.get_features(x)-anogan_D.get_features(xh))**2, dim=1)
            scores.append(((1-lam)*rec + lam*feat).cpu().numpy())
        if s % (bsz*10) == 0:
            print(f'  AnoGAN scored {s+len(x)}/{len(Xq)}')
    return np.concatenate(scores)

print('\nScoring with AnoGAN...')
anogan_scores = anogan_score(X_test)
print(f'Score range: [{anogan_scores.min():.4f}, {anogan_scores.max():.4f}]')

# ── 5. AnoGAN Evaluation ─────────────────────────────────────────────────────
anogan_auroc = roc_auc_score(y_test, anogan_scores)
anogan_auprc = average_precision_score(y_test, anogan_scores)
p, r, t = precision_recall_curve(y_test, anogan_scores)
f1s = 2*p*r/(p+r+1e-9)
bi = np.argmax(f1s)
anogan_preds = (anogan_scores >= t[bi]).astype(int)
anogan_f1 = f1_score(y_test, anogan_preds)
print(f'\nAnoGAN → AUROC={anogan_auroc:.4f}  AUPRC={anogan_auprc:.4f}  F1={anogan_f1:.4f}')
print(classification_report(y_test, anogan_preds, target_names=['Normal','Fraud']))

# ── 6. BiGAN Architecture ────────────────────────────────────────────────────
LATENT_DIM_B = 32

class BiGAN_Generator(nn.Module):
    def __init__(self, ldim=LATENT_DIM_B, odim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ldim,64),  nn.BatchNorm1d(64),  nn.LeakyReLU(0.2),
            nn.Linear(64,128),   nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Linear(128,256),  nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256,odim), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class BiGAN_Encoder(nn.Module):
    def __init__(self, idim=INPUT_DIM, ldim=LATENT_DIM_B):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(idim,256), nn.LeakyReLU(0.2),
            nn.Linear(256,128),  nn.LeakyReLU(0.2),
            nn.Linear(128,64),   nn.LeakyReLU(0.2),
            nn.Linear(64,ldim)
        )
    def forward(self, x): return self.net(x)

class BiGAN_Discriminator(nn.Module):
    def __init__(self, idim=INPUT_DIM, ldim=LATENT_DIM_B):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(idim+ldim,256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256,128),       nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128,64),        nn.LeakyReLU(0.2),
            nn.Linear(64,1),          nn.Sigmoid()
        )
    def forward(self, x, z): return self.net(torch.cat([x,z], dim=1))

bigan_G = BiGAN_Generator().to(DEVICE)
bigan_E = BiGAN_Encoder().to(DEVICE)
bigan_D = BiGAN_Discriminator().to(DEVICE)

# ── 7. BiGAN Training ────────────────────────────────────────────────────────
# FIX: Slightly higher LR for Encoder/Generator to keep up with Discriminator
opt_GE = optim.Adam(
    list(bigan_G.parameters()) + list(bigan_E.parameters()),
    lr=0.0004, betas=(0.5, 0.999)
)
opt_DB = optim.Adam(bigan_D.parameters(), lr=0.0001, betas=(0.5, 0.999))
bce = nn.BCELoss()
bg_g, bg_d = [], []

print('\nTraining BiGAN...')
for epoch in range(EPOCHS):
    gl, dl = 0, 0
    for (rx,) in train_loader:
        rx = rx.to(DEVICE); bsz = rx.size(0)
        # FIX 2 applied to BiGAN as well
        rl = torch.full((bsz,1), 0.9).to(DEVICE)
        fl = torch.zeros(bsz,1).to(DEVICE)

        # ── Discriminator
        opt_DB.zero_grad()
        with torch.no_grad():
            ze = bigan_E(rx)
            z  = torch.randn(bsz, LATENT_DIM_B).to(DEVICE)
            xg = bigan_G(z)
        d_loss = bce(bigan_D(rx, ze), rl) + bce(bigan_D(xg, z), fl)
        d_loss.backward(); opt_DB.step()

        # ── Generator + Encoder
        # FIX 5: Clear variable reuse — use distinct label tensors
        opt_GE.zero_grad()
        ze2 = bigan_E(rx)
        # Encoder wants real pairs (x,E(x)) to fool D → classified as fake
        enc_loss = bce(bigan_D(rx, ze2), torch.zeros(bsz,1).to(DEVICE))
        z2 = torch.randn(bsz, LATENT_DIM_B).to(DEVICE)
        xg2 = bigan_G(z2)
        # Generator wants fake pairs (G(z),z) to fool D → classified as real
        gen_loss = bce(bigan_D(xg2, z2), torch.ones(bsz,1).to(DEVICE))
        ge_loss = enc_loss + gen_loss
        ge_loss.backward(); opt_GE.step()

        gl += ge_loss.item(); dl += d_loss.item()

    n = len(train_loader)
    bg_g.append(gl/n); bg_d.append(dl/n)
    if (epoch+1) % 20 == 0:
        print(f'  Epoch {epoch+1}/{EPOCHS} D={bg_d[-1]:.4f} GE={bg_g[-1]:.4f}')

plt.figure(figsize=(8,4))
plt.plot(bg_g, label='G+E'); plt.plot(bg_d, label='D')
plt.title('BiGAN Training Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout(); plt.savefig('bigan_training_loss.png', dpi=150); plt.show()

# ── 8. BiGAN Inference ───────────────────────────────────────────────────────
bigan_G.eval(); bigan_E.eval(); bigan_D.eval()

def bigan_score(Xq, bsz=512):
    """
    Improved BiGAN scoring: 
    Anomaly = Reconstruction Loss ||x - G(E(x))|| + Discriminator Score
    """
    scores = []
    bigan_G.eval(); bigan_E.eval(); bigan_D.eval()
    with torch.no_grad():
        for s in range(0, len(Xq), bsz):
            x = torch.FloatTensor(Xq[s:s+bsz]).to(DEVICE)
            # Encode and then Decode
            ze = bigan_E(x)
            xh = bigan_G(ze)
            
            # 1. Reconstruction Loss (L1)
            rec_loss = torch.mean(torch.abs(x - xh), dim=1).cpu().numpy()
            
            # 2. Discriminator Score (1 - D)
            d_score = bigan_D(x, ze).squeeze().cpu().numpy()
            
            # Combine them (standard approach for BiGAN anomaly detection)
            # We weight reconstruction higher as it's more stable for tabular data
            combined = 0.9 * rec_loss + 0.1 * (1 - d_score)
            scores.append(combined)
    return np.concatenate(scores)

print('\nScoring with BiGAN...')
bigan_scores = bigan_score(X_test)
print(f'Score range: [{bigan_scores.min():.4f}, {bigan_scores.max():.4f}]')

# ── 9. BiGAN Evaluation ──────────────────────────────────────────────────────
bigan_auroc = roc_auc_score(y_test, bigan_scores)
bigan_auprc = average_precision_score(y_test, bigan_scores)
pb, rb, tb = precision_recall_curve(y_test, bigan_scores)
f1sb = 2*pb*rb/(pb+rb+1e-9)
bib = np.argmax(f1sb)
bigan_preds = (bigan_scores >= tb[bib]).astype(int)
bigan_f1 = f1_score(y_test, bigan_preds)
print(f'\nBiGAN  → AUROC={bigan_auroc:.4f}  AUPRC={bigan_auprc:.4f}  F1={bigan_f1:.4f}')
print(classification_report(y_test, bigan_preds, target_names=['Normal','Fraud']))

# ── 10. Comparison Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ROC
ax = axes[0,0]
for nm, sc, auc in [('AnoGAN',anogan_scores,anogan_auroc),('BiGAN',bigan_scores,bigan_auroc)]:
    fpr,tpr,_ = roc_curve(y_test, sc)
    ax.plot(fpr, tpr, label=f'{nm} AUC={auc:.3f}', lw=2)
ax.plot([0,1],[0,1],'k--'); ax.set_title('ROC Curves')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(); ax.grid(0.3)

# PR
ax = axes[0,1]
for nm, sc, ap in [('AnoGAN',anogan_scores,anogan_auprc),('BiGAN',bigan_scores,bigan_auprc)]:
    p2,r2,_ = precision_recall_curve(y_test, sc)
    ax.plot(r2, p2, label=f'{nm} AP={ap:.3f}', lw=2)
ax.set_title('Precision-Recall'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(); ax.grid(0.3)

# Bar
ax = axes[0,2]
mx = np.arange(3); w = 0.3
ax.bar(mx-w/2,[anogan_auroc,anogan_auprc,anogan_f1],w,label='AnoGAN',color='steelblue')
ax.bar(mx+w/2,[bigan_auroc,bigan_auprc,bigan_f1],  w,label='BiGAN',  color='tomato')
ax.set_xticks(mx); ax.set_xticklabels(['AUROC','AUPRC','F1'])
ax.set_ylim(0,1); ax.set_title('Metric Comparison'); ax.legend(); ax.grid(0.3,axis='y')

# Confusion AnoGAN
sns.heatmap(confusion_matrix(y_test,anogan_preds), annot=True, fmt='d',
            cmap='Blues', ax=axes[1,0],
            xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
axes[1,0].set_title('AnoGAN Confusion Matrix')

# Confusion BiGAN
sns.heatmap(confusion_matrix(y_test,bigan_preds), annot=True, fmt='d',
            cmap='Reds', ax=axes[1,1],
            xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
axes[1,1].set_title('BiGAN Confusion Matrix')

# Score distributions
ax = axes[1,2]
ni, fi = y_test==0, y_test==1
ax.hist(anogan_scores[ni],bins=60,alpha=0.5,density=True,label='AnoGAN Normal',color='steelblue')
ax.hist(anogan_scores[fi],bins=60,alpha=0.5,density=True,label='AnoGAN Fraud',color='navy')
ax2 = ax.twinx()
ax2.hist(bigan_scores[ni],bins=60,alpha=0.4,density=True,label='BiGAN Normal',color='salmon')
ax2.hist(bigan_scores[fi],bins=60,alpha=0.4,density=True,label='BiGAN Fraud',color='red')
ax.set_title('Score Distributions')
l1,lb1 = ax.get_legend_handles_labels()
l2,lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=7)

plt.suptitle('Fraud Detection: AnoGAN vs BiGAN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('All plots saved.')

# ── 11. t-SNE Latent Space ───────────────────────────────────────────────────
from sklearn.manifold import TSNE
idx = np.random.choice(len(X_test), 2000, replace=False)
Xs, ys = X_test[idx], y_test[idx]
bigan_E.eval()
with torch.no_grad():
    zs = bigan_E(torch.FloatTensor(Xs).to(DEVICE)).cpu().numpy()
z2d = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(zs)
plt.figure(figsize=(8,6))
plt.scatter(z2d[ys==0,0], z2d[ys==0,1], c='steelblue', alpha=0.4, s=10, label='Normal')
plt.scatter(z2d[ys==1,0], z2d[ys==1,1], c='red', alpha=0.9, s=30, marker='*', label='Fraud')
plt.title('t-SNE of BiGAN Latent Space'); plt.legend()
plt.tight_layout(); plt.savefig('tsne_latent_space.png', dpi=150); plt.show()

# ── 12. Summary ──────────────────────────────────────────────────────────────
# FIX 4: Corrected f-string — was int(y_test==0).sum() which is wrong
print('\n' + '='*60)
print('FINAL SUMMARY')
print('='*60)
print(f"Dataset: {len(df):,} transactions | "
      f"Normal: {(y==0).sum():,} | Fraud: {(y==1).sum():,} | "
      f"Fraud rate: {y.mean()*100:.3f}%")
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  "
      f"(normal={(y_test==0).sum():.0f}, fraud={y_test.sum():.0f})")
print(f"\n{'Model':<12}{'AUROC':>8}{'AUPRC':>8}{'F1':>8}")
print('-'*38)
print(f"{'AnoGAN':<12}{anogan_auroc:>8.4f}{anogan_auprc:>8.4f}{anogan_f1:>8.4f}")
print(f"{'BiGAN':<12}{bigan_auroc:>8.4f}{bigan_auprc:>8.4f}{bigan_f1:>8.4f}")
