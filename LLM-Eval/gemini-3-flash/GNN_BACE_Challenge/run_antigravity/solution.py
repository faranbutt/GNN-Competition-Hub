import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
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

# 1. Load Data
DATA_DIR = 'data/public'
OUTPUT_DIR = 'run_antigravity'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
train_nodes = pd.read_csv(os.path.join(DATA_DIR, 'train_nodes.csv'))
train_edges = pd.read_csv(os.path.join(DATA_DIR, 'train_edges.csv'))
train_labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
test_nodes = pd.read_csv(os.path.join(DATA_DIR, 'test_nodes.csv'))
test_edges = pd.read_csv(os.path.join(DATA_DIR, 'test_edges.csv'))

def build_data_list(nodes, edges, labels=None):
    data_list = []
    # Group by graph_id
    grouped_nodes = nodes.groupby('graph_id')
    grouped_edges = edges.groupby('graph_id')
    
    graph_ids = nodes['graph_id'].unique()
    
    for gid in graph_ids:
        # Node features
        n_df = grouped_nodes.get_group(gid).sort_values('node_id')
        x = torch.tensor(n_df[['nf_0', 'nf_1', 'nf_2', 'nf_3', 'nf_4', 'nf_5', 'nf_6', 'nf_7']].values, dtype=torch.float)
        
        # Edges
        if gid in grouped_edges.groups:
            e_df = grouped_edges.get_group(gid)
            edge_index = torch.tensor(e_df[['src', 'dst']].values.T, dtype=torch.long)
            # Edge features (optional, using ef_0 if present)
            if 'ef_0' in e_df.columns:
                edge_attr = torch.tensor(e_df[['ef_0']].values, dtype=torch.float)
            else:
                edge_attr = None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None
            
        # Label
        y = None
        if labels is not None:
            l_row = labels[labels['id'] == gid]
            if not l_row.empty:
                y = torch.tensor([l_row['target'].values[0]], dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, id=gid)
        data_list.append(data)
    
    return data_list

print("Preparing datasets...")
train_data_list = build_data_list(train_nodes, train_edges, train_labels)
test_data_list = build_data_list(test_nodes, test_edges)

# 2. Model Architecture
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=False)
        
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        
        # 2. Readout layer (Global Pooling)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 3. Classifier
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

# 3. Training and Validation
# Split train_data_list into train and val
labels = [d.y.item() for d in train_data_list]
train_idx, val_idx = train_test_split(range(len(train_data_list)), test_size=0.2, stratify=labels, random_state=42)

train_subset = [train_data_list[i] for i in train_idx]
val_subset = [train_data_list[i] for i in val_idx]

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

# Calculate pos_weight for imbalance
pos_count = sum(labels)
neg_count = len(labels) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float)

device = torch.device('cpu') # Per requirements
model = GNNModel(in_channels=8, hidden_channels=64, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

epochs = 100
best_val_f1 = 0
best_threshold = 0.5

print("Starting training...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    train_loss = total_loss / len(train_subset)
    
    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            val_preds.append(torch.sigmoid(out.view(-1)).cpu().numpy())
            val_targets.append(data.y.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    # Optimize threshold for Macro F1
    thresholds = np.linspace(0.1, 0.9, 17)
    best_epoch_f1 = 0
    best_epoch_threshold = 0.5
    for t in thresholds:
        f1 = f1_score(val_targets, (val_preds > t).astype(int), average='macro')
        if f1 > best_epoch_f1:
            best_epoch_f1 = f1
            best_epoch_threshold = t
            
    if best_epoch_f1 > best_val_f1:
        best_val_f1 = best_epoch_f1
        best_threshold = best_epoch_threshold
        torch.save(model.state_dict(), 'best_model.pt')
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {train_loss:.4f}, Val Macro F1: {best_epoch_f1:.4f} (at t={best_epoch_threshold:.2f})")

print(f"Best Val Macro F1: {best_val_f1:.4f} at threshold {best_threshold:.2f}")

# 4. Generate Predictions
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
test_preds = []
test_ids = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        test_preds.append(torch.sigmoid(out.view(-1)).cpu().numpy())
        test_ids.extend(data.id)

test_preds = np.concatenate(test_preds)
test_binary = (test_preds > best_threshold).astype(int)

submission = pd.DataFrame({'id': test_ids, 'target': test_binary})
submission.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
print(f"Predictions saved to {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

# Save summary
val_acc = accuracy_score(val_targets, (val_preds > best_threshold).astype(int))
summary = pd.DataFrame({
    'val_f1': [best_val_f1],
    'val_acc': [val_acc],
    'threshold': [best_threshold],
    'epochs': [epochs],
    'splits': ['stratified 80/20']
})
summary.to_csv(os.path.join(OUTPUT_DIR, 'run_summary.csv'), index=False)
print(f"Summary saved to {os.path.join(OUTPUT_DIR, 'run_summary.csv')}")
