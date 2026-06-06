import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(GNNModel, self).__init__()
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        return self.post_mp(x)

def train_model():
    set_seed(42)
    device = torch.device('cpu')
    
    # Load data
    train_df = pd.read_csv('data/public/train.csv')
    test_df = pd.read_csv('data/public/test.csv')
    edges_df = pd.read_csv('data/public/graph_edges.csv')
    node_types_df = pd.read_csv('data/public/node_types.csv')
    test_nodes_df = pd.read_csv('data/public/test_nodes.csv')
    
    # Gene expression columns (all except sample_id, node_id, and disease_labels)
    gene_cols = [col for col in train_df.columns if col not in ['sample_id', 'node_id', 'disease_labels']]
    
    # Create unified node list and mapping
    all_nodes = node_types_df['node_id'].values
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    
    # Features
    # Initialize features with zeros
    X = np.zeros((num_nodes, len(gene_cols)))
    
    # Fill cfRNA features
    for _, row in train_df.iterrows():
        idx = node_to_idx[row['node_id']]
        X[idx] = row[gene_cols].values
        
    # Fill placenta features
    for _, row in test_df.iterrows():
        idx = node_to_idx[row['node_id']]
        X[idx] = row[gene_cols].values
        
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float)
    
    # Labels and masks
    y = torch.zeros(num_nodes, dtype=torch.float)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    for _, row in train_df.iterrows():
        idx = node_to_idx[row['node_id']]
        y[idx] = row['disease_labels']
        train_mask[idx] = True
        
    for _, row in test_df.iterrows():
        idx = node_to_idx[row['node_id']]
        test_mask[idx] = True
        
    # Edge index
    src_idx = [node_to_idx[n] for n in edges_df['src']]
    dst_idx = [node_to_idx[n] for n in edges_df['dst']]
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    
    # Cross-validation
    train_indices = torch.where(train_mask)[0].numpy()
    train_labels = y[train_mask].numpy()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_indices))
    test_preds_total = np.zeros(num_nodes)
    
    metrics_per_fold = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_indices, train_labels)):
        print(f"--- Fold {fold+1} ---")
        
        # Real indices in the full X matrix
        real_train_idx = train_indices[train_idx]
        real_val_idx = train_indices[val_idx]
        
        model = GNNModel(in_channels=X.shape[1], hidden_channels=256, out_channels=1, dropout=0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_f1 = 0
        best_fold_preds = None
        
        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            out = model(X, edge_index).squeeze()
            loss = criterion(out[real_train_idx], y[real_train_idx])
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                out = model(X, edge_index).squeeze()
                val_probs = torch.sigmoid(out[real_val_idx]).numpy()
                val_preds = (val_probs > 0.5).astype(int)
                val_f1 = f1_score(y[real_val_idx].numpy(), val_preds)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_fold_preds = torch.sigmoid(out).numpy()
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}")
                
        oof_preds[val_idx] = best_fold_preds[real_val_idx]
        test_preds_total += best_fold_preds
        
    test_preds_avg = test_preds_total / 5
    
    # Optimize threshold
    best_threshold = 0.5
    best_f1 = 0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(train_labels, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    final_val_acc = accuracy_score(train_labels, (oof_preds > best_threshold).astype(int))
    print(f"\nBest Threshold: {best_threshold:.2f}, Best Val F1: {best_f1:.4f}, Acc: {final_val_acc:.4f}")
    
    # Final predictions for test set
    test_final_probs = test_preds_avg[test_mask]
    test_final_preds = (test_final_probs > best_threshold).astype(int)
    
    # Prepare submission
    test_node_ids = all_nodes[test_mask]
    submission_df = pd.DataFrame({
        'id': test_node_ids,
        'y_pred': test_final_preds
    })
    
    # Reorder based on test_nodes.csv
    submission_df = pd.merge(test_nodes_df, submission_df, left_on='id', right_on='id', how='left')
    submission_df.to_csv('run_antigravity/predictions.csv', index=False)
    print("Saved predictions.csv")
    
    # Save summary
    summary_df = pd.DataFrame([{
        'val_f1': best_f1,
        'val_acc': final_val_acc,
        'threshold': best_threshold,
        'epochs': 150,
        'splits': 5
    }])
    summary_df.to_csv('run_antigravity/run_summary.csv', index=False)
    print("Saved run_summary.csv")

if __name__ == "__main__":
    start_time = time.time()
    train_model()
    print(f"Total time: {time.time() - start_time:.2f} seconds")
