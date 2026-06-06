import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from sklearn.metrics import accuracy_score, f1_score

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class GossipCopDataset:
    def __init__(self, split='train', data_dir='data/public'):
        self.split = split
        self.data_dir = data_dir
        
        # Load graph structure
        print(f"[load] Loading {split} graph structure...")
        self.edges = np.loadtxt(os.path.join(data_dir, "A.txt"), delimiter=",", dtype=int)
        self.node_graph_id = np.load(os.path.join(data_dir, "node_graph_id.npy"))
        self.graph_ids = np.load(os.path.join(data_dir, f"{self.split}_idx.npy"))
        
        if self.split != 'test':
            labels_df = pd.read_csv(os.path.join(data_dir, f"{self.split}_labels.csv"))
            self.label_map = dict(zip(labels_df["id"], labels_df["y_true"]))
        
        # Load and concatenate features
        print("[load] Loading and concatenating features...")
        self.node_features = self._load_features()
        print(f"[load] Feature shape: {self.node_features.shape}")

    def _load_features(self):
        def load_npz(name):
            f = np.load(os.path.join(self.data_dir, name))
            sparse = csr_matrix((f['data'], f['indices'], f['indptr']), shape=f['shape'])
            return sparse.toarray()

        feat_bert = load_npz("new_bert_feature.npz")
        feat_spacy = load_npz("new_spacy_feature.npz")
        feat_profile = load_npz("new_profile_feature.npz")
        
        combined = np.concatenate([feat_bert, feat_spacy, feat_profile], axis=1)
        return torch.tensor(combined, dtype=torch.float)

    def build_graph(self, g_id):
        nodes = np.where(self.node_graph_id == g_id)[0]
        mask = np.isin(self.edges[:, 0], nodes) & np.isin(self.edges[:, 1], nodes)
        edge_index_raw = self.edges[mask]
        
        # Remap nodes to 0..N-1
        node_map = {node: i for i, node in enumerate(nodes)}
        edge_index = np.array([[node_map[u], node_map[v]] for u, v in edge_index_raw]).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # If no edges, ensure edge_index is correct shape
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        x = self.node_features[nodes]
        
        data = Data(x=x, edge_index=edge_index)
        data.graph_id = torch.tensor([g_id], dtype=torch.long)
        
        if self.split != 'test':
            data.y = torch.tensor([self.label_map[g_id]], dtype=torch.float)
            
        return data

    def get_loader(self, batch_size=64, shuffle=True):
        print(f"[prep] Building {self.split} dataset...")
        dataset = [self.build_graph(g) for g in self.graph_ids]
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- Model Architecture ---
class ImprovedGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super(ImprovedGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        
        self.lin_news = nn.Linear(in_channels, hidden_channels)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, batch):
        # GNN layers
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index).relu()
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv3(h, edge_index).relu()
        
        # Graph-level pooling (Mean + Max)
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        
        # Root node (news article) feature
        # In this dataset, the root node is the first node of each graph
        # Finding root indices in the batch
        root_indices = []
        current_batch = -1
        for i, b in enumerate(batch):
            if b > current_batch:
                root_indices.append(i)
                current_batch = b
        root_indices = torch.tensor(root_indices, device=x.device)
        
        news_feat = x[root_indices]
        news_out = self.lin_news(news_feat).relu()
        
        # Concatenate pooling and news feature
        combined = torch.cat([h_mean, h_max, news_out], dim=1)
        out = self.fc(combined)
        return torch.sigmoid(out)

# --- Training Loop ---
def train():
    # Load loaders
    train_loader = GossipCopDataset(split='train').get_loader(batch_size=64, shuffle=True)
    val_loader = GossipCopDataset(split='val').get_loader(batch_size=64, shuffle=False)
    
    in_channels = train_loader.dataset[0].num_features
    model = ImprovedGNN(in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    best_val_acc = 0
    epochs = 50
    patience = 10
    trigger_times = 0
    
    print("[train] Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch).view(-1)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch).view(-1)
                all_preds.append(out.cpu())
                all_labels.append(data.y.cpu())
        
        y_pred_probs = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        y_pred = (y_pred_probs >= 0.5).float()
        
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred)
        
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"[train] Early stopping at epoch {epoch}")
                break
                
    return best_val_acc, val_f1, epoch

# --- Inference ---
def inference():
    test_loader = GossipCopDataset(split='test').get_loader(batch_size=64, shuffle=False)
    in_channels = test_loader.dataset[0].num_features
    model = ImprovedGNN(in_channels).to(device)
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    
    all_preds = []
    all_ids = []
    
    print("[test] Starting inference...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch).view(-1)
            all_preds.extend(out.cpu().numpy())
            all_ids.extend(data.graph_id.cpu().numpy())
            
    # We need hard labels for predictions.csv per baseline format
    y_pred_hard = [1.0 if p >= 0.5 else 0.0 for p in all_preds]
    
    df = pd.DataFrame({"id": all_ids, "y_pred": y_pred_hard})
    df.to_csv("predictions.csv", index=False)
    print("[done] predictions.csv saved.")
    return y_pred_hard

if __name__ == "__main__":
    best_acc, best_f1, last_epoch = train()
    inference()
    
    # Save summary
    summary = pd.DataFrame({
        "val_acc": [best_acc],
        "val_f1": [best_f1],
        "threshold": [0.5],
        "epochs": [last_epoch],
        "splits": ["provided"]
    })
    summary.to_csv("run_summary.csv", index=False)
