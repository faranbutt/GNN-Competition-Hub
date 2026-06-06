import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import random
import os

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# Model architecture
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling (Mean + Max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Output layers
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

def build_dataloader(df, batch_size=32, shuffle=False, has_labels=True):
    graph_list = []
    for _, row in df.iterrows():
        # Convert node features
        x = torch.tensor(np.vstack(row["node_feat"]).astype(np.float32))
        
        # Convert edge index (ensure [2, E] shape)
        edge_index = torch.tensor(np.vstack(row["edge_index"]).astype(np.int64))
        
        # Convert edge attributes (if needed, but GCNConv doesn't use them directly)
        edge_attr = torch.tensor(np.vstack(row["edge_attr"]).astype(np.float32))
        
        if has_labels:
            y = torch.tensor(row["label"], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        graph_list.append(data)
    
    return DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)

def main():
    # Paths
    train_path = "data/public/train_data.parquet"
    test_path = "data/public/test_data.parquet"
    
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Split training data for validation
    train_data, val_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
    )
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_df)}")
    
    # Dataloaders
    batch_size = 64
    train_loader = build_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(test_df, batch_size=batch_size, shuffle=False, has_labels=False)
    
    # Model setup
    num_node_features = 527
    num_classes = 2
    hidden_channels = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GNNModel(num_node_features, hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 50 # Reduced to ensure completion within 60 mins on CPU
    best_val_f1 = 0
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        
        val_f1 = f1_score(y_true, y_pred, average='macro')
        val_acc = accuracy_score(y_true, y_pred)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pt")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    print(f"Training finished. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # Load best model for testing
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    
    # Inference on test set
    print("Generating predictions...")
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())
    
    # Save predictions
    submission = pd.DataFrame({'id': test_df['id'], 'y_pred': test_preds})
    submission.to_csv("run_antigravity/predictions.csv", index=False)
    print("Predictions saved to run_antigravity/predictions.csv")
    
    # Save summary
    summary = pd.DataFrame([{
        'val_f1': best_val_f1,
        'val_acc': val_acc, # Final val acc
        'threshold': 0.5,
        'epochs': epochs,
        'splits': '80/20 stratified'
    }])
    summary.to_csv("run_antigravity/run_summary.csv", index=False)
    print("Run summary saved to run_antigravity/run_summary.csv")

if __name__ == "__main__":
    main()
