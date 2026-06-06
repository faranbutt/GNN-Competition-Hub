import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class ResGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = support + torch.mm(adj, support)
        return output + self.bias


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = ResGCNLayer(nfeat, nhid)
        self.gc2 = ResGCNLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)
        return x


def load_data():
    train_path = "data/public/train_graph_free.pkl"
    test_path = "data/public/test_graph_free.pkl"

    if not os.path.exists(train_path):
        base_dir = "/Users/mac/Desktop/comps/gnn-parkinsons-challenge"
        train_path = os.path.join(base_dir, train_path)
        test_path = os.path.join(base_dir, test_path)

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    num_nodes = train_data["num_nodes"]
    edge_index = train_data["edge_index"]
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    rowsum = adj.sum(1)
    d_inv = torch.pow(rowsum, -1.0).flatten()
    d_inv[torch.isinf(d_inv)] = 0.0
    adj = torch.diag(d_inv) @ adj
    return train_data, test_data, adj


def train():
    train_data, test_data, adj = load_data()
    features, labels = train_data["features"], train_data["labels"]
    train_mask, val_mask = train_data["train_mask"], train_data["val_mask"]
    test_node_ids = test_data["node_ids"]

    train_labels = labels[train_mask]
    w0 = len(train_labels) / (2 * (train_labels == 0).sum().item())
    w1 = len(train_labels) / (2 * (train_labels == 1).sum().item())

    model = GCN(nfeat=features.shape[1], nhid=64, nclass=2, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    best_val_f1 = 0
    best_state, best_info = None, {}

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.cross_entropy(
            output[train_mask], labels[train_mask], weight=torch.FloatTensor([w0, w1])
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            val_preds = output[val_mask].max(1)[1]
            val_f1 = f1_score(
                labels[val_mask].numpy(), val_preds.numpy(), average="macro"
            )
            val_acc = (val_preds == labels[val_mask]).float().mean().item()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_info = {"epoch": epoch + 1, "val_f1": val_f1, "val_acc": val_acc}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(features, adj)[test_node_ids].max(1)[1]

    pd.DataFrame({"node_id": test_node_ids, "prediction": test_preds.numpy()}).to_csv(
        "run_gemini-3-flash/predictions.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "val_f1": best_info["val_f1"],
                "val_acc": best_info["val_acc"],
                "threshold": 0.5,
                "epochs": best_info["epoch"],
                "splits": "default",
            }
        ]
    ).to_csv("run_gemini-3-flash/run_summary.csv", index=False)
    print(f"Final Best Val F1: {best_info['val_f1']:.4f}")


if __name__ == "__main__":
    train()
