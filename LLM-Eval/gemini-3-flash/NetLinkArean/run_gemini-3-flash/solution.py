import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    def __init__(self):
        super(LinkPredictor, self).__init__()

    def forward(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def main():
    set_seed(42)
    device = torch.device("cpu")

    # 1. Load Data
    print("[load] loading data...")
    node_features_df = pd.read_csv("data/public/node_features.csv").sort_values(
        "node_id"
    )
    x = torch.tensor(
        node_features_df.drop(columns=["node_id"]).values, dtype=torch.float
    )
    # L2 Normalization
    x = F.normalize(x, p=2, dim=-1)

    train_edges_df = pd.read_csv("data/public/train_edges.csv")
    val_edges_df = pd.read_csv("data/public/val_edges.csv")
    test_nodes_df = pd.read_csv("data/public/test_nodes.csv")

    # 2. Graph Construction (Message Passing edges from positive training edges)
    pos_train_edges = train_edges_df[train_edges_df["label"] == 1]
    G = nx.Graph()
    G.add_nodes_from(range(x.size(0)))
    for _, row in pos_train_edges.iterrows():
        G.add_edge(int(row["source"]), int(row["target"]))

    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Make it undirected explicitly for PyG
    edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    # 3. Prepare Training and Validation Pairs
    train_src = torch.tensor(train_edges_df["source"].values, dtype=torch.long)
    train_dst = torch.tensor(train_edges_df["target"].values, dtype=torch.long)
    train_labels = torch.tensor(train_edges_df["label"].values, dtype=torch.float)

    val_src = torch.tensor(val_edges_df["source"].values, dtype=torch.long)
    val_dst = torch.tensor(val_edges_df["target"].values, dtype=torch.long)
    val_labels = val_edges_df["label"].values

    # 4. Initialize Model
    in_channels = x.size(1)
    hidden_channels = 128
    out_channels = 64
    encoder = GNNEncoder(in_channels, hidden_channels, out_channels).to(device)
    predictor = LinkPredictor().to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0
    best_epoch = 0
    epochs = 150
    patience = 30
    counter = 0

    print(f"[train] features: {x.shape}, MP edges: {edge_index.shape[1]}")

    for epoch in range(1, epochs + 1):
        encoder.train()
        optimizer.zero_grad()

        z = encoder(x, edge_index)

        # Training scores
        logits = predictor(z, (train_src, train_dst))
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Validation
        encoder.eval()
        with torch.no_grad():
            z = encoder(x, edge_index)
            val_logits = predictor(z, (val_src, val_dst))
            val_probs = torch.sigmoid(val_logits).numpy()
            val_auc = roc_auc_score(val_labels, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save(encoder.state_dict(), "best_encoder.pt")
            counter = 0
        else:
            counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}"
            )

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"[done] best val_auc={best_val_auc:.4f} at epoch {best_epoch}")

    # 6. Inference
    encoder.load_state_dict(torch.load("best_encoder.pt"))
    encoder.eval()
    with torch.no_grad():
        z = encoder(x, edge_index)
        test_src = torch.tensor(test_nodes_df["source"].values, dtype=torch.long)
        test_dst = torch.tensor(test_nodes_df["target"].values, dtype=torch.long)
        test_logits = predictor(z, (test_src, test_dst))
        test_probs = torch.sigmoid(test_logits).numpy()

    # 7. Save Submission
    submission = pd.DataFrame({"id": test_nodes_df["id"], "y_pred": test_probs})
    submission.to_csv("predictions.csv", index=False)
    print(f"[write] predictions.csv saved ({len(submission)} rows)")

    # 8. Save Summary
    summary = pd.DataFrame(
        [
            {
                "val_auc": best_val_auc,
                "val_f1": 0.0,  # Not requested but placeholder
                "val_acc": 0.0,  # Not requested but placeholder
                "threshold": 0.5,
                "epochs": best_epoch,
                "splits": "provided",
            }
        ]
    )
    summary.to_csv("run_summary.csv", index=False)


if __name__ == "__main__":
    main()
