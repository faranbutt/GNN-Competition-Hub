import os
import pickle
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# GCN Model using manual sparse matrix multiplication
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        return torch.spmm(adj, x)


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.gcn2 = GCNLayer(hidden_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.gcn3 = GCNLayer(hidden_features, hidden_features)
        self.bn3 = nn.BatchNorm1d(hidden_features)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x, adj):
        x = F.relu(self.bn1(self.gcn1(x, adj)))
        x = F.relu(self.bn2(self.gcn2(x, adj)))
        x = F.relu(self.bn3(self.gcn3(x, adj)))

        # Global Pooling (Mean + Max)
        x_mean = torch.mean(x, dim=0)
        x_max = torch.max(x, dim=0)[0]
        x_pool = torch.cat([x_mean, x_max], dim=0)

        return self.classifier(x_pool.unsqueeze(0))


def load_graph_tensors(file_path):
    try:
        with open(file_path, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

    nodes = list(G.nodes())
    num_nodes = len(nodes)
    if num_nodes == 0:
        return None, None

    mapping = {node: i for i, node in enumerate(nodes)}

    # Node Features: x, y coordinates and degree
    xs = np.array([G.nodes[n].get("x", 0) for n in nodes], dtype=np.float32)
    ys = np.array([G.nodes[n].get("y", 0) for n in nodes], dtype=np.float32)

    # Center and scale coordinates
    xs = (xs - xs.mean()) / (xs.std() + 1e-6)
    ys = (ys - ys.mean()) / (ys.std() + 1e-6)

    degrees = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)

    X = np.stack([xs, ys, degrees], axis=1)
    X = torch.tensor(X, dtype=torch.float32)

    # Adjacency with self-loops
    edges = list(G.edges())
    indices = []
    for u, v in edges:
        if u in mapping and v in mapping:
            indices.append([mapping[u], mapping[v]])
            indices.append([mapping[v], mapping[u]])
    # Self loops
    for i in range(num_nodes):
        indices.append([i, i])

    indices = torch.tensor(indices).t()
    values = torch.ones(indices.shape[1])

    # Symmetric Normalization: D^-0.5 * A * D^-0.5
    row, col = indices
    deg = torch.zeros(num_nodes).scatter_add_(0, row, values)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        indices, norm_values, (num_nodes, num_nodes)
    ).coalesce()

    return X, adj


def main():
    TRAIN_DIR = "data/train"
    TEST_DIR = "data/test"
    LABELS_CSV = "data/train_labels.csv"

    # Load labels
    labels_df = pd.read_csv(LABELS_CSV)
    train_filenames = labels_df["filename"].tolist()
    train_labels = labels_df["target"].tolist()

    print(f"Loading {len(train_filenames)} training graphs...")
    train_graphs = []
    valid_labels = []
    for fn, label in zip(train_filenames, train_labels):
        X, adj = load_graph_tensors(os.path.join(TRAIN_DIR, fn))
        if X is not None:
            train_graphs.append((X, adj))
            valid_labels.append(label)

    y = torch.tensor(valid_labels, dtype=torch.long)

    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_models = []
    fold_metrics = []

    # Class weights for loss
    class_counts = np.bincount(valid_labels)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(train_graphs)), valid_labels)
    ):
        print(f"\nTraining Fold {fold + 1}...")
        model = GNNModel(in_features=3, hidden_features=64, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1 = 0
        best_model_state = None
        patience = 50
        counter = 0

        for epoch in range(300):
            model.train()
            train_loss = 0
            # Shuffle train index
            np.random.shuffle(train_idx)
            for idx in train_idx:
                X, adj = train_graphs[idx]
                optimizer.zero_grad()
                out = model(X, adj)
                loss = criterion(out, y[idx].unsqueeze(0))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for idx in val_idx:
                    X, adj = train_graphs[idx]
                    out = model(X, adj)
                    val_preds.append(torch.argmax(out, dim=1).item())
                    val_targets.append(y[idx].item())

            val_f1 = f1_score(val_targets, val_preds, average="macro")
            val_acc = accuracy_score(val_targets, val_preds)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = pickle.loads(pickle.dumps(model.state_dict()))
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                break

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}"
                )

        print(f"Fold {fold + 1} Best Val F1: {best_val_f1:.4f}")
        model.load_state_dict(best_model_state)
        fold_models.append(model)

        # Final val metrics
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for idx in val_idx:
                X, adj = train_graphs[idx]
                out = model(X, adj)
                val_preds.append(torch.argmax(out, dim=1).item())
                val_targets.append(y[idx].item())

        final_f1 = f1_score(val_targets, val_preds, average="macro")
        final_acc = accuracy_score(val_targets, val_preds)
        fold_metrics.append((final_f1, final_acc))

    avg_f1 = np.mean([m[0] for m in fold_metrics])
    avg_acc = np.mean([m[1] for m in fold_metrics])
    print(f"\nCV Average F1: {avg_f1:.4f}, Average Acc: {avg_acc:.4f}")

    # Inference on test set
    print("\nInference on test set...")
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".pkl")])
    test_results = []

    for fn in test_files:
        X, adj = load_graph_tensors(os.path.join(TEST_DIR, fn))
        if X is None:
            # Fallback to most frequent class if loading fails (usually hybrid=2)
            test_results.append({"filename": fn, "prediction": 2})
            continue

        # Ensemble predictions
        fold_probs = []
        for model in fold_models:
            model.eval()
            with torch.no_grad():
                out = model(X, adj)
                probs = F.softmax(out, dim=1)
                fold_probs.append(probs)

        avg_probs = torch.mean(torch.stack(fold_probs), dim=0)
        pred = torch.argmax(avg_probs, dim=1).item()
        test_results.append({"filename": fn, "prediction": pred})

    # Save submission
    submission_df = pd.DataFrame(test_results)
    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

    # Save summary for run_summary.csv
    summary = {
        "val_f1": avg_f1,
        "val_acc": avg_acc,
        "threshold": "N/A",
        "epochs": 300,
        "splits": 5,
    }
    pd.DataFrame([summary]).to_csv("run_summary.csv", index=False)


if __name__ == "__main__":
    main()
