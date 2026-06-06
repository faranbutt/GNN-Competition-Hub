import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import os
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

DATA_DIR = 'data/public'
OUTPUT_DIR = 'run_antigravity'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
print("Loading graphs and features...")
adj = sp.load_npz(os.path.join(DATA_DIR, 'adjacency_matrix.npz'))
feats = np.load(os.path.join(DATA_DIR, 'node_features.npy'))
train_target = pd.read_csv(os.path.join(DATA_DIR, 'train_target.csv'))
test_target_without_labels = pd.read_csv(os.path.join(DATA_DIR, 'test_target_without_labels.csv'))

# Process adjacency into edge index
coo = adj.tocoo()
edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
x = torch.tensor(feats, dtype=torch.float)

# Maps to map ID to index and label
num_nodes = x.size(0)
y = torch.zeros(num_nodes, dtype=torch.float)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# Stratified split for train/val
labels = train_target['ml_target'].values
train_ids, val_ids, y_train, y_val = train_test_split(
    train_target['id'].values, labels, test_size=0.2, stratify=labels, random_state=42
)

# Populate masks
train_set = set(train_ids)
val_set = set(val_ids)
for _, row in train_target.iterrows():
    nid = int(row['id'])
    y[nid] = row['ml_target']
    if nid in train_set:
        train_mask[nid] = True
    elif nid in val_set:
        val_mask[nid] = True

test_node_ids = test_target_without_labels['id'].values
for nid in test_node_ids:
    test_mask[nid] = True

# 2. Model Definition
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.lin(x)
        return x

device = torch.device('cpu')
model = GNNModel(in_channels=x.size(1), hidden_channels=128, out_channels=1).to(device)

# Pos weight calculation
train_labels_tensor = y[train_mask]
pos_count = train_labels_tensor.sum().item()
neg_count = len(train_labels_tensor) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 3. Training Loop
epochs = 150
best_val_f1 = 0
best_threshold = 0.5
best_model_path = 'best_octo_model.pt'

print("Starting training...")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask].view(-1), y[train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(x, edge_index)[val_mask].view(-1)
        val_probs = torch.sigmoid(val_logits).numpy()
        val_targets = y[val_mask].numpy()
        
        # Search for best threshold on validation
        thresholds = np.linspace(0.2, 0.8, 13)
        best_epoch_f1 = 0
        best_epoch_threshold = 0.5
        for t in thresholds:
            preds = (val_probs > t).astype(int)
            f1 = f1_score(val_targets, preds, average='macro')
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_threshold = t
                
        if best_epoch_f1 > best_val_f1:
            best_val_f1 = best_epoch_f1
            best_threshold = best_epoch_threshold
            torch.save(model.state_dict(), best_model_path)
            
    if epoch % 10 == 0:
        val_acc = accuracy_score(val_targets, (val_probs > best_epoch_threshold).astype(int))
        print(f"Epoch {epoch:03d}: Loss={loss.item():.4f}, Val F1={best_epoch_f1:.4f} (at t={best_epoch_threshold:.2f}), Val Acc={val_acc*100:.1f}%")

print(f"Best Val Macro F1: {best_val_f1:.4f} at threshold {best_threshold:.2f}")

# 4. Predictions on Test
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    out = model(x, edge_index)
    test_logits = out[test_mask].view(-1)
    test_probs = torch.sigmoid(test_logits).numpy()
    test_preds = (test_probs > best_threshold).astype(int)

# Formulate predictions matching test_target_without_labels exactly
test_target_without_labels['ml_target'] = test_preds

# Save predictions.csv
submission = test_target_without_labels[['id', 'name', 'ml_target']]
submission.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
print(f"Predictions saved to {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

# Save run summary
val_acc = accuracy_score(val_targets, (val_probs > best_threshold).astype(int))
summary = pd.DataFrame({
    'val_f1': [best_val_f1],
    'val_acc': [val_acc],
    'threshold': [best_threshold],
    'epochs': [epochs],
    'splits': ['stratified 80/20']
})
summary.to_csv(os.path.join(OUTPUT_DIR, 'run_summary.csv'), index=False)
print(f"Summary saved to {os.path.join(OUTPUT_DIR, 'run_summary.csv')}")
