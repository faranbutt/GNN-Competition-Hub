import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
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

# 1. Load Data
DATA_DIR = 'data/public'
OUTPUT_DIR = 'run_antigravity'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
nodes_df = pd.read_csv(os.path.join(DATA_DIR, 'nodes.csv'))
edges_df = pd.read_csv(os.path.join(DATA_DIR, 'edges.csv'))
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test_nodes_df = pd.read_csv(os.path.join(DATA_DIR, 'test_nodes.csv'))

# Preprocess features
# Fill NaN (HbA1c_level) with 0 as flag exists
nodes_df = nodes_df.fillna(0)

# Map id to index
id_to_idx = {id_val: i for i, id_val in enumerate(nodes_df['id'])}

# Node features
feat_cols = [c for c in nodes_df.columns if c != 'id']
x = torch.tensor(nodes_df[feat_cols].values, dtype=torch.float)

# Edges
# Map src and dst to indices
src_idx = edges_df['src'].map(id_to_idx).values
dst_idx = edges_df['dst'].map(id_to_idx).values
edge_index = torch.tensor(np.array([src_idx, dst_idx]), dtype=torch.long)

# Masks and Labels
num_nodes = len(nodes_df)
y = torch.zeros(num_nodes, dtype=torch.float)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

for _, row in train_df.iterrows():
    idx = id_to_idx[row['id']]
    y[idx] = row['diabetes']
    train_mask[idx] = True

for _, row in val_df.iterrows():
    idx = id_to_idx[row['id']]
    y[idx] = row['diabetes']
    val_mask[idx] = True

test_ids = test_nodes_df['id'].values
for tid in test_ids:
    idx = id_to_idx[tid]
    test_mask[idx] = True

# 2. Model Architecture
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

# 3. Training
device = torch.device('cpu')
model = GNNModel(in_channels=len(feat_cols), hidden_channels=128, out_channels=1).to(device)

# Calculate pos_weight
train_labels = y[train_mask]
pos_count = train_labels.sum().item()
neg_count = len(train_labels) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_f1 = 0
epochs = 150

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
        val_preds = (val_probs > 0.5).astype(int)
        
        f1 = f1_score(val_targets, val_preds, average='macro')
        acc = accuracy_score(val_targets, val_preds)
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), 'best_model_dia.pt')
            
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Val Macro F1: {f1:.4f}, Val Acc: {acc:.4f}")

print(f"Best Val Macro F1: {best_val_f1:.4f}")

# 4. Generate Predictions
model.load_state_dict(torch.load('best_model_dia.pt'))
model.eval()
with torch.no_grad():
    out = model(x, edge_index)
    test_logits = out[test_mask].view(-1)
    test_probs = torch.sigmoid(test_logits).numpy()

submission = pd.DataFrame({'id': test_ids, 'y_pred': test_probs})
submission.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
print(f"Predictions saved to {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

# Save summary
summary = pd.DataFrame({
    'val_f1': [best_val_f1],
    'val_acc': [acc], # Last epoch acc or best? Using acc from current context for simplicity
    'threshold': [0.5],
    'epochs': [epochs],
    'splits': ['provided train/val']
})
summary.to_csv(os.path.join(OUTPUT_DIR, 'run_summary.csv'), index=False)
print(f"Summary saved to {os.path.join(OUTPUT_DIR, 'run_summary.csv')}")
