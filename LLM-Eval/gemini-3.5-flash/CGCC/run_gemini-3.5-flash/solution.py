import os
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# ----------------------------------------------------
# 1. PATHS AND SEED SETUP
# ----------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"Using device: {DEVICE}")

# ----------------------------------------------------
# 2. FEATURE ENGINEERING FUNCTIONS
# ----------------------------------------------------
def one_hot_street_count(sc: int) -> np.ndarray:
    arr = np.zeros(5, dtype=np.float32)
    if sc <= 1:
        arr[0] = 1.0
    elif sc == 2:
        arr[1] = 1.0
    elif sc == 3:
        arr[2] = 1.0
    elif sc == 4:
        arr[3] = 1.0
    else:
        arr[4] = 1.0
    return arr

def compute_bearing_entropy(G: nx.Graph) -> float:
    angles = []
    for u, v in G.edges():
        ux, uy = G.nodes[u].get('x', 0.0), G.nodes[u].get('y', 0.0)
        vx, vy = G.nodes[v].get('x', 0.0), G.nodes[v].get('y', 0.0)
        dx = vx - ux
        dy = vy - uy
        if dx != 0.0 or dy != 0.0:
            angles.append(np.arctan2(dy, dx))
    
    if not angles:
        return 0.0
        
    angles_mod = np.array(angles) % (np.pi / 2.0)
    hist, _ = np.histogram(angles_mod, bins=10, range=(0.0, np.pi/2.0))
    hist = hist / max(1, hist.sum())
    entropy = -np.sum(hist * np.log(hist + 1e-9))
    return float(entropy)

def build_node_features(G: nx.Graph) -> np.ndarray:
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return None

    xs = np.array([G.nodes[n].get("x", 0.0) for n in nodes], dtype=np.float32)
    ys = np.array([G.nodes[n].get("y", 0.0) for n in nodes], dtype=np.float32)

    xs = xs - xs.mean()
    ys = ys - ys.mean()

    scale = float(np.sqrt(xs.var() + ys.var()) + 1e-6)
    xs = xs / scale
    ys = ys / scale

    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)

    # 1. Local clustering coefficient
    clustering_dict = nx.clustering(nx.Graph(G))
    clusts = np.array([clustering_dict.get(n, 0.0) for n in nodes], dtype=np.float32)

    # 2. One-hot street counts
    scs = np.array([one_hot_street_count(G.nodes[n].get("street_count", 0)) for n in nodes], dtype=np.float32)

    # Concat features: x (1), y (1), degree (1), clustering (1), street_count (5) => 9 dims
    return np.hstack([xs[:, None], ys[:, None], deg[:, None], clusts[:, None], scs])

def build_global_features(G: nx.Graph) -> np.ndarray:
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    edge_node_ratio = num_edges / max(1, num_nodes)
    
    degrees = [G.degree(n) for n in G.nodes]
    deg_mean = np.mean(degrees) if degrees else 0.0
    deg_std = np.std(degrees) if degrees else 0.0
    
    street_counts = [G.nodes[n].get("street_count", 0) for n in G.nodes]
    sc_mean = np.mean(street_counts) if street_counts else 0.0
    sc_std = np.std(street_counts) if street_counts else 0.0
    
    pct_sc3 = np.mean([1.0 if c == 3 else 0.0 for c in street_counts]) if street_counts else 0.0
    pct_sc4 = np.mean([1.0 if c == 4 else 0.0 for c in street_counts]) if street_counts else 0.0
    
    lengths = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if "length" in d:
            lengths.append(d["length"])
    
    len_mean = np.mean(lengths) if lengths else 0.0
    len_std = np.std(lengths) if lengths else 0.0
    len_cv = len_std / max(1e-6, len_mean)
    
    entropy_mod = compute_bearing_entropy(G)
    
    return np.array([
        num_nodes * 0.01,
        num_edges * 0.01,
        edge_node_ratio,
        deg_mean,
        deg_std,
        sc_mean,
        sc_std,
        pct_sc3,
        pct_sc4,
        len_mean * 0.01,
        len_cv,
        entropy_mod
    ], dtype=np.float32)

def coalesce_undirected_edges(G: nx.Graph, mapping: dict):
    edges = []
    for u, v in G.edges():
        if u in mapping and v in mapping:
            iu, iv = mapping[u], mapping[v]
            edges.append((iu, iv))
            edges.append((iv, iu))
    return edges

def normalize_adj_sparse(indices: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
    indices = indices.long()
    values = values.float()

    row = indices[0]
    deg = torch.zeros(n, device=values.device).scatter_add_(0, row, values)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

    col = indices[1]
    norm_values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(indices, norm_values, (n, n)).coalesce()

def graph_to_tensors(G: nx.Graph, device: torch.device):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return None, None, None

    mapping = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    X_np = build_node_features(G)
    if X_np is None:
        return None, None, None
    X = torch.tensor(X_np, dtype=torch.float32, device=device)

    edges = coalesce_undirected_edges(G, mapping)
    edges += [(i, i) for i in range(n)] # Add self-loops
    if len(edges) == 0:
        return None, None, None

    indices = torch.tensor(edges, dtype=torch.long, device=device).t()
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    A_norm = normalize_adj_sparse(adj.indices(), adj.values(), n)
    
    g_np = build_global_features(G)
    g = torch.tensor(g_np, dtype=torch.float32, device=device).unsqueeze(0) # [1, 12]

    return X, A_norm, g

# ----------------------------------------------------
# 3. DATA LOADING FUNCTIONS
# ----------------------------------------------------
def load_train_data(train_dir: str, labels_csv: str, device: torch.device):
    labels_df = pd.read_csv(labels_csv)
    label_map = dict(zip(labels_df["filename"], labels_df["target"]))

    files = sorted([f for f in os.listdir(train_dir) if f.endswith(".pkl")])
    graphs, y = [], []

    for fn in files:
        if fn not in label_map:
            continue
        with open(os.path.join(train_dir, fn), "rb") as f:
            G = pickle.load(f)

        X, A, g = graph_to_tensors(G, device)
        if X is None:
            continue

        graphs.append((fn, X, A, g))
        y.append(int(label_map[fn]))

    return graphs, torch.tensor(y, dtype=torch.long, device=device)

def load_test_data(test_dir: str, device: torch.device):
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".pkl")])
    graphs = []

    for fn in files:
        with open(os.path.join(test_dir, fn), "rb") as f:
            G = pickle.load(f)

        X, A, g = graph_to_tensors(G, device)
        if X is None:
            continue

        graphs.append((fn, X, A, g))

    return graphs

# ----------------------------------------------------
# 4. GNN MODEL ARCHITECTURE
# ----------------------------------------------------
class ResidualGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, adj):
        h = torch.spmm(adj, self.linear(x))
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h + self.shortcut(x)

class GNNFusionModel(nn.Module):
    def __init__(self, node_in_dim=9, global_in_dim=12, hidden_dim=64, num_classes=3, dropout=0.4):
        super().__init__()
        self.gcn1 = ResidualGCNLayer(node_in_dim, hidden_dim, dropout=dropout)
        self.gcn2 = ResidualGCNLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in_dim, 32),
            nn.ReLU(),
            self.gcn1.bn.__class__(32), # Use same normalization class
            nn.Dropout(dropout)
        )
        
        fused_dim = hidden_dim * 2 + 32
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, adj, g):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        
        h_mean = h.mean(dim=0)
        h_max = h.max(dim=0).values
        h_pool = torch.cat([h_mean, h_max], dim=0) # [hidden_dim * 2]
        
        h_global = self.global_mlp(g).squeeze(0) # [32]
        
        h_fused = torch.cat([h_pool, h_global], dim=0).unsqueeze(0) # [1, fused_dim]
        return self.classifier(h_fused)

# ----------------------------------------------------
# 5. MODEL TRAINING & CROSS-VALIDATION
# ----------------------------------------------------
train_graphs, y_all = load_train_data(TRAIN_DIR, TRAIN_LABELS_CSV, DEVICE)
test_graphs = load_test_data(TEST_DIR, DEVICE)

print(f"Loaded train graphs: {len(train_graphs)}")
print(f"Loaded test graphs:  {len(test_graphs)}")

labels_cpu = y_all.detach().cpu().numpy()
num_samples = len(labels_cpu)
idx_all = np.arange(num_samples)

# Compute class weights for balanced loss
counts = np.bincount(labels_cpu, minlength=3)
print(f"Class distribution: {counts}")
weights = counts.sum() / (counts + 1e-6)
weights = weights / weights.mean()
class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

# 5-fold cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_val_accuracies = []
fold_val_f1_scores = []
models = []

max_epochs = 200
patience = 40
accum_steps = 4

print("\n--- Starting 5-Fold Cross-Validation ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(idx_all, labels_cpu)):
    print(f"\n>>> Fold {fold + 1}/5")
    
    # Instantiate model, optimizer, scheduler, criterion
    model = GNNFusionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_f1 = -1.0
    best_state = None
    bad_epochs = 0
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Shuffle train indices for batch size = 1 gradient accumulation
        perm = np.random.permutation(train_idx)
        optimizer.zero_grad()
        
        for step, idx in enumerate(perm):
            _, X, A, g = train_graphs[idx]
            target = y_all[idx].unsqueeze(0)
            
            logits = model(X, A, g)
            loss = criterion(logits, target) / accum_steps
            loss.backward()
            train_loss += float(loss.item()) * accum_steps
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(perm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        scheduler.step()
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for idx in val_idx:
                _, X, A, g = train_graphs[idx]
                target = y_all[idx].unsqueeze(0)
                
                logits = model(X, A, g)
                loss = criterion(logits, target)
                val_loss += float(loss.item())
                
                pred = int(torch.argmax(logits, dim=1).item())
                y_true.append(int(target.item()))
                y_pred.append(pred)
                
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={train_loss/len(train_idx):.4f} | "
                  f"val_loss={val_loss/len(val_idx):.4f} | val_acc={val_acc:.3f} | val_f1={val_f1:.3f}")
                  
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val F1: {best_val_f1:.4f}")
                break
                
    # Load best model weights and calculate final fold metrics
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in val_idx:
            _, X, A, g = train_graphs[idx]
            logits = model(X, A, g)
            pred = int(torch.argmax(logits, dim=1).item())
            y_true.append(int(y_all[idx].item()))
            y_pred.append(pred)
            
    final_acc = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred, average="macro")
    
    fold_val_accuracies.append(final_acc)
    fold_val_f1_scores.append(final_f1)
    models.append(model)
    
    print(f"Fold {fold+1} finished: Final Val Acc = {final_acc:.4f}, Final Val F1 = {final_f1:.4f}")

# Calculate CV stats
mean_cv_acc = np.mean(fold_val_accuracies)
mean_cv_f1 = np.mean(fold_val_f1_scores)
print("\n=== CROSS-VALIDATION SUMMARY ===")
print(f"Mean CV Accuracy: {mean_cv_acc:.4f}")
print(f"Mean CV Macro-F1: {mean_cv_f1:.4f}")

# ----------------------------------------------------
# 6. ENSEMBLED INFERENCE ON TEST SET
# ----------------------------------------------------
print("\n--- Running Ensembled Inference on Test Set ---")
pred_rows = []

for fn, X, A, g in test_graphs:
    # Average soft probabilities across all fold models
    soft_probs = torch.zeros(1, 3, device=DEVICE)
    
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(X, A, g)
            soft_probs += F.softmax(logits, dim=1)
            
    soft_probs /= len(models)
    pred = int(torch.argmax(soft_probs, dim=1).item())
    pred_rows.append({"filename": fn, "prediction": pred})

submission = pd.DataFrame(pred_rows).sort_values("filename")
out_path = os.path.join(SCRIPT_DIR, "predictions.csv")
submission.to_csv(out_path, index=False)
print(f"Wrote predictions to: {out_path}")
print(submission.head())

# ----------------------------------------------------
# 7. WRITE RUN SUMMARY
# ----------------------------------------------------
summary_path = os.path.join(SCRIPT_DIR, "run_summary.csv")
summary_df = pd.DataFrame([{
    "val_f1": f"{mean_cv_f1:.4f}",
    "val_acc": f"{mean_cv_acc:.4f}",
    "threshold": "N/A (Softmax argmax)",
    "epochs": f"{max_epochs}",
    "splits": "5-fold CV"
}])
summary_df.to_csv(summary_path, index=False)
print(f"Wrote run summary to: {summary_path}")
