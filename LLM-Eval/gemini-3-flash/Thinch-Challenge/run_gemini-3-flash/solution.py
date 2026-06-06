import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import f1_score

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def load_data():
    # Use relative paths from repo root
    train_df = pd.read_csv('data/public/train.csv')
    edge_df = pd.read_csv('data/public/edge_list.csv')
    test_nodes_df = pd.read_csv('data/public/test_nodes.csv')
    test_edges_df = pd.read_csv('data/public/test_edges.csv')

    # Feature columns
    feat_cols = ['x', 'y', 'width', 'height']

    # Train Data
    train_id_map = {row_id: i for i, row_id in enumerate(train_df['id'].values)}
    train_x = torch.tensor(train_df[feat_cols].values, dtype=torch.float32)
    train_y = torch.tensor(train_df['label'].values, dtype=torch.long)

    train_src = [train_id_map[i] for i in edge_df['source']]
    train_dst = [train_id_map[i] for i in edge_df['target']]
    train_edge_index = torch.tensor([train_src, train_dst], dtype=torch.long)

    train_data = Data(x=train_x, y=train_y, edge_index=train_edge_index)

    # Test Data
    test_id_map = {row_id: i for i, row_id in enumerate(test_nodes_df['id'].values)}
    test_x = torch.tensor(test_nodes_df[feat_cols].values, dtype=torch.float32)

    test_src = [test_id_map[i] for i in test_edges_df['source']]
    test_dst = [test_id_map[i] for i in test_edges_df['target']]
    test_edge_index = torch.tensor([test_src, test_dst], dtype=torch.long)

    test_data = Data(x=test_x, edge_index=test_edge_index)
    test_ids = test_nodes_df['id'].values

    return train_data, test_data, test_ids

def train():
    train_data, test_data, test_ids = load_data()

    # Random split for validation
    num_nodes = train_data.x.size(0)
    indices = list(range(num_nodes))
    random.shuffle(indices)
    split = int(num_nodes * 0.8)
    train_idx = torch.tensor(indices[:split])
    val_idx = torch.tensor(indices[split:])

    # Class weights
    train_labels = train_data.y[train_idx]
    class_counts = np.bincount(train_labels.numpy(), minlength=4)
    class_weights = torch.tensor(len(train_labels) / (4.0 * class_counts), dtype=torch.float32)

    model = GCN(in_feats=4, h_feats=64, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_f1 = 0
    best_state = None
    best_info = {}

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        loss = F.cross_entropy(out[train_idx], train_data.y[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            out = model(train_data.x, train_data.edge_index)
            val_preds = out[val_idx].argmax(dim=1)
            val_f1 = f1_score(train_data.y[val_idx].numpy(), val_preds.numpy(), average='macro')
            val_acc = (val_preds == train_data.y[val_idx]).float().mean().item()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_info = {'epoch': epoch + 1, 'val_f1': val_f1, 'val_acc': val_acc}

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f}")

    print(f"Best Validation F1: {best_val_f1:.4f} at epoch {best_info['epoch']}")

    # Predict on test data
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index)
        test_preds = out.argmax(dim=1)

    submission = pd.DataFrame({
        'id': test_ids,
        'y_pred': test_preds.numpy()
    })

    output_dir = 'run_gemini-3-flash'
    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(f'{output_dir}/predictions.csv', index=False)
    print(f"Predictions saved to {output_dir}/predictions.csv")

    # Save summary
    summary = pd.DataFrame([{
        'val_f1': best_info['val_f1'],
        'val_acc': best_info['val_acc'],
        'threshold': 0.5,
        'epochs': best_info['epoch'],
        'splits': 'random_80_20'
    }])
    summary.to_csv(f'{output_dir}/run_summary.csv', index=False)

if __name__ == '__main__':
    train()
