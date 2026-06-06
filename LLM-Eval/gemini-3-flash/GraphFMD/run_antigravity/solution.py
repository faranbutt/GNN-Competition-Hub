import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# Set seeds for reproducibility
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

# GNN Model Definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def train():
    # Paths
    base_path = "/Users/mac/Desktop/comps/GraphFMD"
    data_path = os.path.join(base_path, "data/public")
    
    # Load data
    print("Loading data...")
    train_nodes = pd.read_csv(os.path.join(data_path, 'train_nodes.csv'))
    train_labels = pd.read_csv(os.path.join(data_path, 'train_labels.csv'))
    test_nodes = pd.read_csv(os.path.join(data_path, 'test_nodes.csv'))
    edgelist = pd.read_csv(os.path.join(data_path, 'edgelist.csv'))
    
    # Combine nodes for feature processing and mapping
    all_nodes = pd.concat([train_nodes, test_nodes], axis=0).reset_index(drop=True)
    
    # Map IDs to indices
    node_id_map = {node_id: i for i, node_id in enumerate(all_nodes['id'])}
    
    # Features
    # Column 0 is id, Column 1 is time_step, others are features
    X_raw = all_nodes.iloc[:, 2:].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = torch.tensor(X_scaled, dtype=torch.float)
    
    # Edges
    print("Constructing edge index...")
    source_nodes = edgelist['txId1'].map(node_id_map)
    target_nodes = edgelist['txId2'].map(node_id_map)
    
    # Filter out edges where nodes are not in our set
    valid_mask = source_nodes.notna() & target_nodes.notna()
    edge_index = torch.tensor([
        source_nodes[valid_mask].astype(int).values,
        target_nodes[valid_mask].astype(int).values
    ], dtype=torch.long)
    
    # Labels
    # y=1 (Illicit) -> class 1
    # y=2 (Licit) -> class 0
    train_labels['y_mapped'] = train_labels['y'].map({1: 1, 2: 0})
    
    # Split training labels into train and validation
    train_idx_full, val_idx_full = train_test_split(
        train_labels.index, test_size=0.2, stratify=train_labels['y_mapped'], random_state=42
    )
    
    train_label_nodes = train_labels.loc[train_idx_full, 'id'].map(node_id_map).values
    train_y = train_labels.loc[train_idx_full, 'y_mapped'].values
    
    val_label_nodes = train_labels.loc[val_idx_full, 'id'].map(node_id_map).values
    val_y = train_labels.loc[val_idx_full, 'y_mapped'].values
    
    # Full labels for all nodes (-1 for unlabeled/test)
    Y = torch.full((len(all_nodes),), -1, dtype=torch.long)
    Y[train_label_nodes] = torch.tensor(train_y, dtype=torch.long)
    
    # Class weights for imbalance
    class_counts = train_labels.loc[train_idx_full, 'y_mapped'].value_counts()
    weight_0 = 1.0 / class_counts[0]
    weight_1 = 1.0 / class_counts[1]
    # Normalize weights
    total = weight_0 + weight_1
    weights = torch.tensor([weight_0/total, weight_1/total], dtype=torch.float)
    
    # Device
    device = torch.device('cpu') # Forced CPU as per prompt
    X = X.to(device)
    edge_index = edge_index.to(device)
    Y = Y.to(device)
    weights = weights.to(device)
    
    # Model
    model = GCN(in_channels=X.shape[1], hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Training loop
    epochs = 100
    print(f"Starting training for {epochs} epochs...")
    best_val_f1 = 0
    best_epoch = 0
    best_val_acc = 0
    
    output_dir = "/Users/mac/Desktop/comps/GraphFMD/run_antigravity"
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X, edge_index)
        loss = criterion(out[train_label_nodes], Y[train_label_nodes])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X, edge_index)
            val_logits = logits[val_label_nodes]
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            
            val_f1 = f1_score(val_y, val_preds, average='macro')
            val_acc = accuracy_score(val_y, val_preds)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_epoch = epoch
                # Save best model state
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Best Epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f}")
    
    # Load best model for prediction
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    model.eval()
    
    # Prediction on test set
    test_ids = test_nodes['id'].values
    test_idx = [node_id_map[tid] for tid in test_ids]
    
    with torch.no_grad():
        logits = model(X, edge_index)
        test_logits = logits[test_idx]
        test_preds = test_logits.argmax(dim=1).cpu().numpy()
        
    # Map back to competition format: class 1 -> y=1 (Illicit), class 0 -> y=2 (Licit)
    final_preds = np.where(test_preds == 1, 1, 2)
    
    # Save predictions
    submission = pd.DataFrame({'id': test_ids, 'y_pred': final_preds})
    submission.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print("Predictions saved to predictions.csv")
    
    # Save run summary
    summary = pd.DataFrame({
        'val_f1': [best_val_f1],
        'val_acc': [best_val_acc],
        'threshold': [0.5],
        'epochs': [epochs],
        'splits': ['80/20 Stratified']
    })
    summary.to_csv(os.path.join(output_dir, 'run_summary.csv'), index=False)
    print("Summary saved to run_summary.csv")

if __name__ == "__main__":
    train()
