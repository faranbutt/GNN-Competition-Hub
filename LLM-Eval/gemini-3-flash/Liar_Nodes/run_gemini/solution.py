import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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
data_dir = "/Users/mac/Desktop/comps/Liar-Nodes-Challenge/data"
edges_path = os.path.join(data_dir, "edges.csv")
labels_path = os.path.join(data_dir, "labels.csv")
test_path = os.path.join(data_dir, "test.csv")
train_path = os.path.join(data_dir, "train_compressed.csv")

# 3. Load data
print("Loading CSV files...")
edges = pd.read_csv(edges_path)
labels = pd.read_csv(labels_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

train_ids = train['Unnamed: 0'].values
test_ids = test['Unnamed: 0'].values

train_feats = train.drop(columns=['Unnamed: 0', 'is_perturbed']).values
test_feats = test.drop(columns=['Unnamed: 0']).values
train_mask = train['is_perturbed'].values
train_y = labels['cell_type'].values

# 4. Perturbation mask predictor
print("Training Random Forest to predict perturbation mask...")
rf_mask = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_mask.fit(train_feats, train_mask)
test_mask_pred = rf_mask.predict(test_feats)

# Combine masks and features into 4,700-node full graph
print("Assembling transductive graph features...")
all_feats = np.zeros((4700, train_feats.shape[1]), dtype=np.float32)
all_feats[train_ids] = train_feats
all_feats[test_ids] = test_feats

all_mask = np.zeros(4700, dtype=np.float32)
all_mask[train_ids] = train_mask
all_mask[test_ids] = test_mask_pred

# Build edge index for full graph
edge_index = torch.tensor(edges[['source', 'target']].values.T, dtype=torch.long)
edge_weight = torch.tensor(edges['weight'].values, dtype=torch.float)

y_full = np.zeros(4700, dtype=np.int64)
y_full[train_ids] = train_y

x_tensor = torch.tensor(all_feats, dtype=torch.float)
y_tensor = torch.tensor(y_full, dtype=torch.long)
mask_tensor = torch.tensor(all_mask, dtype=torch.float)

data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running GNN training on device:", device)

# 5. Define Custom Masked SAGE GNN
class MaskedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, mask):
        h_neigh = self.propagate(edge_index, x=x)
        h_self = self.lin_self(x)
        # Multiply self features by (1.0 - mask) to ignore corrupted self-features
        h_self = h_self * (1.0 - mask.view(-1, 1))
        return h_self + self.lin_neigh(h_neigh)

    def message(self, x_j):
        return x_j

class MaskedSAGEGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = MaskedSAGEConv(in_channels, hidden_channels)
        self.conv2 = MaskedSAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, mask):
        x = self.conv1(x, edge_index, mask)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, mask)
        return x

# 6. Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_accs = []
fold_f1s = []
test_probs_all_folds = []

epochs = 150
lr = 0.01
weight_decay = 5e-4

print(f"Starting 5-Fold Cross-Validation ({epochs} epochs per fold)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_ids, train_y)):
    print(f"\n--- Fold {fold + 1} / 5 ---")
    
    # Map back to full graph indices
    train_nodes = train_ids[train_idx]
    val_nodes = train_ids[val_idx]
    
    # Initialize model, optimizer, criterion
    model = MaskedSAGEGNN(in_channels=x_tensor.shape[1], hidden_channels=128, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    data_device = data.to(device)
    mask_device = mask_tensor.to(device)
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_device.x, data_device.edge_index, mask_device)
        loss = criterion(out[train_nodes], data_device.y[train_nodes])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(data_device.x, data_device.edge_index, mask_device)
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            val_true = train_y[val_idx]
            val_preds = preds[val_nodes]
            
            val_acc = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average='macro')
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
    print(f"Fold {fold + 1} Finished! Best Val Acc: {best_val_acc:.4f} | Best Val F1: {best_val_f1:.4f}")
    fold_accs.append(best_val_acc)
    fold_f1s.append(best_val_f1)
    
    # Predict on test set using the best model checkpoint
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data_device.x, data_device.edge_index, mask_device)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        test_probs = probs[test_ids] # extract test node probabilities
        test_probs_all_folds.append(test_probs)

mean_val_acc = np.mean(fold_accs)
mean_val_f1 = np.mean(fold_f1s)
print(f"\n======================================")
print(f"Overall 5-Fold Cross-Validation Metrics:")
print(f"Mean Val Accuracy: {mean_val_acc:.6f}")
print(f"Mean Val F1: {mean_val_f1:.6f}")
print(f"======================================")

# 7. Ensemble & Save predictions
print("\nEnsembling predictions across all folds...")
avg_test_probs = np.mean(test_probs_all_folds, axis=0)
final_test_preds = np.argmax(avg_test_probs, axis=1)

# Align predictions with test.csv Unnamed: 0 order
submission = pd.DataFrame({
    'Unnamed: 0': test_ids,
    'cell_type': final_test_preds
})

out_dir = "/Users/mac/Desktop/comps/Liar-Nodes-Challenge/run_gemini"
os.makedirs(out_dir, exist_ok=True)

pred_path = os.path.join(out_dir, "predictions.csv")
submission.to_csv(pred_path, index=False)
print(f"Predictions saved to: {pred_path}")

# 8. Save run summary csv
summary_path = os.path.join(out_dir, "run_summary.csv")
summary_df = pd.DataFrame([{
    'val_f1': f"{mean_val_f1:.6f}",
    'val_acc': f"{mean_val_acc:.6f}",
    'threshold': 'argmax',
    'epochs': str(epochs),
    'splits': '5-fold Stratified'
}])
summary_df.to_csv(summary_path, index=False)
print(f"Summary metrics saved to: {summary_path}")

print("solution.py execution completed successfully!")
