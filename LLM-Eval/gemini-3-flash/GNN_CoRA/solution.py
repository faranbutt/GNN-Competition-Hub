import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Set seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_data():
    print("Loading data...")
    # Load features
    x = pd.read_csv("data/x.csv").values.astype(np.float32)
    x = torch.from_numpy(x)

    # Load edges
    edge_index_df = pd.read_csv("data/edge_index.csv")
    # The CSV has source, target columns. PyG expects [2, E]
    edge_index = torch.from_numpy(edge_index_df.values.T).long()

    # Load labels
    y_train_df = pd.read_csv("data/y_train.csv")
    y_val_df = pd.read_csv("data/y_val.csv")
    test_id_df = pd.read_csv("data/test_ID.csv")

    num_nodes = x.shape[0]
    y = torch.zeros(num_nodes, dtype=torch.long)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for _, row in y_train_df.iterrows():
        idx, lbl = int(row["index"]), int(row["label"])
        y[idx] = lbl
        train_mask[idx] = True

    for _, row in y_val_df.iterrows():
        idx, lbl = int(row["index"]), int(row["label"])
        y[idx] = lbl
        val_mask[idx] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    return data, test_id_df


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():
    data, test_id_df = load_data()
    num_classes = 7

    model = GCN(num_features=data.num_features, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("Training model...")
    best_val_acc = 0
    best_model_state = None

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        logits = model(data)

        # Train Accuracy
        train_pred = logits[data.train_mask].max(1)[1]
        train_acc = (
            train_pred.eq(data.y[data.train_mask]).sum().item()
            / data.train_mask.sum().item()
        )

        # Val Accuracy
        val_pred = logits[data.val_mask].max(1)[1]
        val_acc = (
            val_pred.eq(data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

    print(f"Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_state)
    model.eval()
    logits = model(data)
    preds = logits.max(1)[1].numpy()

    test_ids = test_id_df["id"].values
    test_preds = preds[test_ids]

    submission = pd.DataFrame({"id": test_ids, "target": test_preds})
    submission.to_csv("predictions.csv", index=False)
    submission.to_csv("submission.csv", index=False)
    print("Predictions saved to predictions.csv and submission.csv")

    print("Predictions saved to predictions.csv and submission.csv")

if __name__ == "__main__":
    main()
