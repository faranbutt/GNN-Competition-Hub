import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 1. Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# 2. Path Resolution for Brain Connectivity Matrices
def get_adj_path(subject, mr_session):
    if str(subject).startswith('Test_Sub'):
        return f"data/public/adjacency_matrices/{subject}-{mr_session}.csv"
    else:
        name = str(mr_session).replace('_MR_', '_')
        return f"data/public/adjacency_matrices/{name}.csv"

# 3. Load and Normalize Adjacency Matrix
def load_adj(adj_path, max_log_val=None):
    adj = pd.read_csv(adj_path, index_col=0).values
    adj = np.nan_to_num(adj)
    adj_log = np.log1p(adj)
    if max_log_val is not None:
        adj_scaled = adj_log / max_log_val
    else:
        adj_scaled = adj_log
    return adj_scaled

def main():
    print("--- 1. LOADING METADATA ---")
    train_df = pd.read_csv("data/public/train_data.csv")
    val_df = pd.read_csv("data/public/val_data.csv")
    test_df = pd.read_csv("data/public/test_data.csv")
    
    # Extract structural cortical measure column names (136 columns)
    feature_cols = [col for col in train_df.columns if col.startswith('ctx-')]
    assert len(feature_cols) == 136, f"Expected 136 features, got {len(feature_cols)}"
    
    print(f"Loaded {len(train_df)} train, {len(val_df)} val, and {len(test_df)} test sessions.")

    print("--- 2. STANDARDIZING NODE FEATURES ---")
    # Region-wise scaling using StandardScaler fit on training features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_df[feature_cols].values)
    val_features_scaled = scaler.transform(val_df[feature_cols].values)
    test_features_scaled = scaler.transform(test_df[feature_cols].values)

    print("--- 3. CALCULATING GLOBAL TRAINING MAXIMUM LOG STREAMLINE COUNT ---")
    max_log_val = 1.0
    for _, row in train_df.iterrows():
        adj_path = get_adj_path(row['Subject'], row['MR_session'])
        if os.path.exists(adj_path):
            adj = pd.read_csv(adj_path, index_col=0).values
            adj_log = np.log1p(adj)
            max_log_val = max(max_log_val, adj_log.max())
    print(f"Global training maximum log streamline count: {max_log_val:.4f}")

    # Helper function to construct PyG Data list
    def build_pyg_data(df, scaled_features, max_log_val, is_test=False):
        data_list = []
        for idx, row in df.iterrows():
            subject = row['Subject']
            mr_session = row['MR_session']
            
            # Label (Chronological Age)
            if not is_test:
                y = torch.tensor([row['age at visit']], dtype=torch.float32)
            else:
                y = None
                
            # Node features of shape (68, 2)
            feat = scaled_features[idx].reshape(68, 2)
            x = torch.tensor(feat, dtype=torch.float32)
            
            # Adjacency matrix loaded and scaled
            adj_path = get_adj_path(subject, mr_session)
            adj_scaled = load_adj(adj_path, max_log_val)
            
            # Convert to PyG edge_index and edge_weight (edge_attr)
            adj_tensor = torch.tensor(adj_scaled, dtype=torch.float32)
            edge_index = adj_tensor.nonzero().t().contiguous()
            edge_weight = adj_tensor[edge_index[0], edge_index[1]]
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y,
                subject_session=f"{subject}-{mr_session}"
            )
            data_list.append(data)
        return data_list

    print("--- 4. CONSTRUCTING PyG GRAPHS ---")
    train_dataset = build_pyg_data(train_df, train_features_scaled, max_log_val, is_test=False)
    val_dataset = build_pyg_data(val_df, val_features_scaled, max_log_val, is_test=False)
    test_dataset = build_pyg_data(test_df, test_features_scaled, max_log_val, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 5. GNN Architecture for Graph-level Brain Regression
    class BrainAgeGNN(nn.Module):
        def __init__(self, in_channels=2, hidden_channels=64):
            super(BrainAgeGNN, self).__init__()
            # Disable add_self_loops to prevent diluting the diagonal self-streamline weights
            self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False)
            self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
            self.conv3 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
            
            self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
            self.fc3 = nn.Linear(hidden_channels // 2, 1)
            
            self.dropout = nn.Dropout(p=0.2)
            
        def forward(self, x, edge_index, edge_weight, batch):
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            
            x = self.conv2(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.dropout(x)
            
            x = self.conv3(x, edge_index, edge_weight)
            x = F.relu(x)
            
            # Global pooling (concatenating mean and max pool)
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            x_pool = torch.cat([mean_pool, max_pool], dim=1)
            
            # Fully connected layers
            x_pool = self.fc1(x_pool)
            x_pool = F.relu(x_pool)
            x_pool = self.dropout(x_pool)
            
            x_pool = self.fc2(x_pool)
            x_pool = F.relu(x_pool)
            
            out = self.fc3(x_pool)
            return out.view(-1)

    print("--- 5. INITIALIZING GNN MODEL ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    
    model = BrainAgeGNN(in_channels=2, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.L1Loss() # MAE Loss directly optimizes the metric

    def train_epoch():
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(train_loader.dataset)

    def eval_model(loader):
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                preds.extend(out.cpu().numpy())
                if batch.y is not None:
                    targets.extend(batch.y.cpu().numpy())
        if len(targets) > 0:
            mae = mean_absolute_error(targets, preds)
            return mae, preds
        return None, preds

    print("--- 6. TRAINING LOOP ---")
    best_val_mae = float('inf')
    best_model_state = None
    epochs = 120

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch()
        val_mae, _ = eval_model(val_loader)
        
        # Keep checkpoint with lowest validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MAE: {train_loss:.4f} | Val MAE: {val_mae:.4f}")

    print(f"Best Validation MAE reached: {best_val_mae:.4f}")

    print("--- 7. EVALUATING BEST MODEL CHECKPOINT ---")
    model.load_state_dict(best_model_state)
    model = model.to(device)
    
    final_val_mae, _ = eval_model(val_loader)
    print(f"Final best validation set MAE: {final_val_mae:.4f}")

    # Generate test-set predictions
    print("--- 8. PREDICTING ON TEST SET ---")
    _, test_preds = eval_model(test_loader)

    # 9. Write outputs in required format
    sub_df = pd.DataFrame({
        'subject_session': [data.subject_session for data in test_dataset],
        'age_at_visit': test_preds
    })
    
    # Save predictions.csv to repo root
    sub_df.to_csv('predictions.csv', index=False)
    # Save predictions.csv to run_gemini/
    sub_df.to_csv('run_gemini/predictions.csv', index=False)
    print("Saved predictions.csv successfully.")

    # Save run_summary.csv in run_gemini/
    summary_df = pd.DataFrame([{
        'val_f1': -1.0,
        'val_acc': -1.0,
        'threshold': -1.0,
        'epochs': epochs,
        'splits': 1,
        'val_mae': final_val_mae
    }])
    summary_df.to_csv('run_gemini/run_summary.csv', index=False)
    print("Saved run_summary.csv successfully.")

if __name__ == '__main__':
    main()
