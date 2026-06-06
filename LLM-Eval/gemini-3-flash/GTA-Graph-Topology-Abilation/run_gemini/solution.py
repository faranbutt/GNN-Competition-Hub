import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# 1. Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("Starting solution.py pipeline...")

# 2. Paths
data_dir = "/Users/mac/Desktop/comps/GTA-Graph-Topology-Ablation/data"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
out_dir = "/Users/mac/Desktop/comps/GTA-Graph-Topology-Ablation/run_gemini"
os.makedirs(out_dir, exist_ok=True)

# 3. Load MUTAG dataset
print("Loading TUDataset MUTAG...")
dataset = TUDataset(root=data_dir, name="MUTAG")

# 4. Load splits
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Extract graph indices and labels
train_indices = train_df['graph_index'].values
train_labels = train_df['label'].values
test_indices = test_df['graph_index'].values

# Load train graphs
train_graphs = []
for idx, label in zip(train_indices, train_labels):
    g = dataset[int(idx)]
    g.y = torch.tensor(int(label), dtype=torch.long)
    train_graphs.append(g)

# Load test graphs
ideal_test_graphs = []
for idx in test_indices:
    g = dataset[int(idx)]
    ideal_test_graphs.append(g)

# Perturbation function
def perturb_graph(data, feature_shift=0.3, noise_std=0.05):
    data = data.clone()
    if data.x is not None:
        shift = torch.full_like(data.x, feature_shift)
        noise = torch.randn_like(data.x) * noise_std
        data.x = data.x + shift + noise
    return data

# Perturbed test graphs
perturbed_test_graphs = [perturb_graph(g) for g in ideal_test_graphs]

# 5. Define robust GIN Model
class GINModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        nn_seq = Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(nn_seq)
        self.lin = Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# 6. Stratified 5-Fold Cross-Validation Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

epochs = 120
lr = 0.005
wd = 5e-4
batch_size = 16

fold_ideal_f1s = []
fold_perturbed_f1s = []

test_ideal_probs_all = []
test_perturbed_probs_all = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running training on device:", device)

print(f"Starting 5-Fold CV GIN training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_indices, train_labels)):
    print(f"\n--- Fold {fold + 1} / 5 ---")
    
    # 50/50 clean and perturbed training data augmentation
    fold_train_graphs = [train_graphs[i] for i in train_idx]
    augmented_train = []
    for g in fold_train_graphs:
        augmented_train.append(g)
        augmented_train.append(perturb_graph(g))
        
    train_loader = DataLoader(augmented_train, batch_size=batch_size, shuffle=True)
    
    # Validation loaders
    fold_val_clean = [train_graphs[i] for i in val_idx]
    fold_val_perturbed = [perturb_graph(train_graphs[i]) for i in val_idx]
    
    val_loader_clean = DataLoader(fold_val_clean, batch_size=32)
    val_loader_perturbed = DataLoader(fold_val_perturbed, batch_size=32)
    
    # Initialize model
    model = GINModel(dataset.num_features, dataset.num_classes, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    best_perturbed_f1 = 0.0
    best_ideal_f1 = 0.0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            out = model(batch_data)
            loss = F.nll_loss(out, batch_data.y)
            loss.backward()
            optimizer.step()
            
        # Evaluate
        model.eval()
        with torch.no_grad():
            # Clean validation
            preds_clean = []
            true_clean = []
            for batch_data in val_loader_clean:
                batch_data = batch_data.to(device)
                out = model(batch_data)
                preds_clean.extend(out.argmax(dim=1).tolist())
                true_clean.extend(batch_data.y.tolist())
            f1_ideal = f1_score(true_clean, preds_clean, average="macro")
            
            # Perturbed validation
            preds_perturbed = []
            true_perturbed = []
            for batch_data in val_loader_perturbed:
                batch_data = batch_data.to(device)
                out = model(batch_data)
                preds_perturbed.extend(out.argmax(dim=1).tolist())
                true_perturbed.extend(batch_data.y.tolist())
            f1_perturbed = f1_score(true_perturbed, preds_perturbed, average="macro")
            
        if f1_perturbed > best_perturbed_f1:
            best_perturbed_f1 = f1_perturbed
            best_ideal_f1 = f1_ideal
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Val F1 Ideal: {f1_ideal:.4f} | Val F1 Perturbed: {f1_perturbed:.4f}")
            
    print(f"Fold {fold + 1} Best Results -> Val F1 Ideal: {best_ideal_f1:.4f} | Val F1 Perturbed: {best_perturbed_f1:.4f}")
    fold_ideal_f1s.append(best_ideal_f1)
    fold_perturbed_f1s.append(best_perturbed_f1)
    
    # Predict on test set using the best fold checkpoint
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    ideal_test_loader = DataLoader(ideal_test_graphs, batch_size=32)
    perturbed_test_loader = DataLoader(perturbed_test_graphs, batch_size=32)
    
    with torch.no_grad():
        # Ideal test
        ideal_probs = []
        for batch_data in ideal_test_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            probs = torch.exp(out) # log_softmax to probs
            ideal_probs.extend(probs.cpu().numpy().tolist())
        test_ideal_probs_all.append(ideal_probs)
        
        # Perturbed test
        perturbed_probs = []
        for batch_data in perturbed_test_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)
            probs = torch.exp(out)
            perturbed_probs.extend(probs.cpu().numpy().tolist())
        test_perturbed_probs_all.append(perturbed_probs)

mean_val_f1_ideal = np.mean(fold_ideal_f1s)
mean_val_f1_perturbed = np.mean(fold_perturbed_f1s)
gap = mean_val_f1_ideal - mean_val_f1_perturbed

print("\n======================================")
print("Overall 5-Fold Cross-Validation Metrics:")
print(f"Mean Val F1 Ideal: {mean_val_f1_ideal:.6f}")
print(f"Mean Val F1 Perturbed: {mean_val_f1_perturbed:.6f}")
print(f"Robustness Gap: {gap:.6f}")
print("======================================")

# 7. Ensemble test set predictions
print("\nEnsembling predictions across all folds...")
avg_ideal_probs = np.mean(test_ideal_probs_all, axis=0)
final_ideal_preds = np.argmax(avg_ideal_probs, axis=1)

avg_perturbed_probs = np.mean(test_perturbed_probs_all, axis=0)
final_perturbed_preds = np.argmax(avg_perturbed_probs, axis=1)

# Align predictions with test_df order and save as graph_index,label
ideal_sub = pd.DataFrame({
    'graph_index': test_indices,
    'label': final_ideal_preds
})

perturbed_sub = pd.DataFrame({
    'graph_index': test_indices,
    'label': final_perturbed_preds
})

ideal_out_path = os.path.join(out_dir, "ideal.csv")
perturbed_out_path = os.path.join(out_dir, "perturbed.csv")

ideal_sub.to_csv(ideal_out_path, index=False)
perturbed_sub.to_csv(perturbed_out_path, index=False)

print(f"Saved clean predictions to: {ideal_out_path}")
print(f"Saved perturbed predictions to: {perturbed_out_path}")

# 8. Save run summary csv
summary_path = os.path.join(out_dir, "run_summary.csv")
summary_df = pd.DataFrame([{
    'val_f1_ideal': f"{mean_val_f1_ideal:.6f}",
    'val_f1_perturbed': f"{mean_val_f1_perturbed:.6f}",
    'robustness_gap': f"{gap:.6f}",
    'epochs': str(epochs)
}])
summary_df.to_csv(summary_path, index=False)
print(f"Summary metrics saved to: {summary_path}")

print("solution.py pipeline finished successfully!")
