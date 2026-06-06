import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import pandas as pd
import numpy as np
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

class GrapeGAT(torch.nn.Module):
    def __init__(self, in_dim, hid=64, out=2, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # GAT layers with multi-head attention
        self.conv1 = GATConv(in_dim, hid, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hid * heads, hid, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hid * heads, hid, heads=1, dropout=dropout)
        
        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(hid * heads)
        self.bn2 = torch.nn.BatchNorm1d(hid * heads)
        self.bn3 = torch.nn.BatchNorm1d(hid)
        
        # MLP classifier (3 pooling strategies * hid + 5 graph features)
        self.fc1 = torch.nn.Linear(hid * 3 + 5, hid)
        self.fc2 = torch.nn.Linear(hid, out)
    
    def forward(self, x, edge_index, batch, graph_feats):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))
        
        # Multiple pooling strategies
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_add = global_add_pool(x, batch)
        
        # Concatenate pooled features with graph-level features
        x = torch.cat([x_mean, x_max, x_add, graph_feats], dim=1)
        
        # MLP classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

def compute_graph_features(g, node_map, edges):
    num_nodes = len(g)
    num_edges = len(edges)
    
    degree = np.zeros(num_nodes)
    for e in edges:
        degree[e[0]] += 1
    
    avg_degree = degree.mean() if num_nodes > 0 else 0
    junction_ratio = (g['type'] == 'junction').mean() if 'type' in g.columns else 0
    avg_width = g['width'].mean() if 'width' in g.columns else 0
    
    return [num_nodes / 500, num_edges / 500, avg_degree / 5, junction_ratio, avg_width / 10]

def load_graphs(graph_path, label_path=None):
    df = pd.read_csv(graph_path)
    labels = pd.read_csv(label_path) if label_path else None
    graphs = []
    
    for gid in df['graph_id'].unique():
        g = df[df['graph_id']==gid].reset_index(drop=True)
        
        node_type_map = {'junction': 0, 'endpoint': 1}
        type_vals = g['type'].map(lambda t: node_type_map.get(t, 0)).values if 'type' in g.columns else np.zeros(len(g))
        
        x = np.column_stack([
            g['x'].values / 600,
            g['y'].values / 600,
            g['width'].values / 20,
            type_vals,
        ])
        x = torch.tensor(x, dtype=torch.float)
        
        edges = []
        node_map = {row['node_id']: i for i, row in g.iterrows()}
        for i, row in g.iterrows():
            if pd.notna(row['edges']) and row['edges']:
                for tgt in str(row['edges']).split(';'):
                    if tgt.strip().isdigit():
                        tgt_id = int(tgt)
                        if tgt_id in node_map:
                            edges.append([i, node_map[tgt_id]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros(2,0,dtype=torch.long)
        graph_feats = compute_graph_features(g, node_map, edges)
        
        y = torch.tensor([labels[labels['graph_id']==gid]['label'].values[0]]) if labels is not None else None
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.gid = gid
        data.graph_feats = torch.tensor([graph_feats], dtype=torch.float)
        graphs.append(data)
        
    return graphs

def train():
    DATA_DIR = 'data/public'
    OUTPUT_DIR = 'run_antigravity'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    graphs = load_graphs(os.path.join(DATA_DIR, 'train_data.csv'), os.path.join(DATA_DIR, 'train_labels.csv'))
    
    # Stratified split based on class labels
    labels = [g.y.item() for g in graphs]
    
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(range(len(graphs)), test_size=0.2, stratify=labels, random_state=42)
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Class weights
    train_labels = [g.y.item() for g in train_graphs]
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * 2
    print(f"Class weights: {class_weights.tolist()}")
    
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    device = torch.device('cpu')
    model = GrapeGAT(in_dim=4, hid=64, out=2, heads=4, dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    
    best_val_f1 = 0
    best_model_state = None
    patience = 40
    patience_counter = 0
    
    print("Starting training...")
    for ep in range(300):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            graph_feats = torch.cat([g.graph_feats for g in batch.to_data_list()], dim=0).to(device)
            out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
            loss = F.cross_entropy(out, batch.y, weight=class_weights.to(device))
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += len(batch.y)
            
        scheduler.step()
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                graph_feats = torch.cat([g.graph_feats for g in batch.to_data_list()], dim=0).to(device)
                out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
                pred = out.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = accuracy_score(val_targets, val_preds)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1:03d}: Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc*100:.1f}%, Val F1={val_f1:.4f}, Val Acc={val_acc*100:.1f}%")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep+1}")
            break
            
    model.load_state_dict(best_model_state)
    print(f"Best Val Macro F1: {best_val_f1:.4f}")
    
    # Save Predictions
    test_graphs = load_graphs(os.path.join(DATA_DIR, 'test_data.csv'))
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            graph_feats = torch.cat([g.graph_feats for g in batch.to_data_list()], dim=0).to(device)
            out = model(batch.x, batch.edge_index, batch.batch, graph_feats)
            pred = out.argmax(dim=1).item()
            preds.append({'graph_id': batch.to_data_list()[0].gid, 'label': pred})
            
    submission = pd.DataFrame(preds)
    submission.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(OUTPUT_DIR, 'predictions.csv')}")
    
    # Save summary
    summary = pd.DataFrame({
        'val_f1': [best_val_f1],
        'val_acc': [val_acc],
        'threshold': [0.5],
        'epochs': [ep+1],
        'splits': ['stratified 80/20']
    })
    summary.to_csv(os.path.join(OUTPUT_DIR, 'run_summary.csv'), index=False)
    print(f"Summary saved to {os.path.join(OUTPUT_DIR, 'run_summary.csv')}")

if __name__ == "__main__":
    train()
