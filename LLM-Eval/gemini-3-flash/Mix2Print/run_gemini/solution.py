import os
import re
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# --- Feature Parsers ---
def parse_components(comp_str):
    if pd.isna(comp_str) or not str(comp_str).strip():
        return []
    pattern = r'([A-Za-z0-9\s\-\(\),]+?)\s*\[([0-9.]+)\s*([a-zA-Z%/]+)\]'
    matches = re.findall(pattern, comp_str)
    components = []
    for name, conc, unit in matches:
        components.append({
            'name': name.strip(),
            'concentration': float(conc),
            'unit': unit.strip()
        })
    return components

def parse_needle(needle_str):
    if pd.isna(needle_str) or not str(needle_str).strip():
        return {'diameter': 400.0, 'geometry': 'unknown'}
    
    needle_str = str(needle_str).strip()
    diameter = None
    
    # Try µm pattern
    um_match = re.search(r'([0-9.]+)\s*[µu]m', needle_str, re.IGNORECASE)
    if um_match:
        diameter = float(um_match.group(1))
    else:
        # Try gauge conversion (approximate)
        gauge_match = re.search(r'([0-9]+)\s*[Gg]auge', needle_str)
        if gauge_match:
            gauge = int(gauge_match.group(1))
            gauge_to_um = {
                18: 838, 19: 686, 20: 603, 21: 514, 22: 413,
                23: 337, 24: 311, 25: 260, 26: 260, 27: 210,
                30: 159, 32: 108
            }
            diameter = gauge_to_um.get(gauge, 400.0)
            
    if diameter is None:
        diameter = 400.0
        
    geometry = 'unknown'
    if 'cylindrical' in needle_str.lower():
        geometry = 'cylindrical'
    elif 'conical' in needle_str.lower():
        geometry = 'conical'
        
    return {'diameter': diameter, 'geometry': geometry}

def parse_cells(cells_str):
    if pd.isna(cells_str) or not str(cells_str).strip():
        return 0.0
    cells_str = str(cells_str).strip()
    # Find numbers inside parentheses first
    match = re.search(r'\(([0-9.]+)\)', cells_str)
    if match:
        return float(match.group(1))
    # Otherwise find any floating point number
    match = re.search(r'([0-9.]+)', cells_str)
    if match:
        return float(match.group(1))
    return 0.0

def get_composition_vector(components, mat_to_idx, num_materials):
    vec = np.zeros(num_materials, dtype=np.float32)
    for comp in components:
        name = comp['name']
        conc = comp['concentration']
        if name in mat_to_idx:
            vec[mat_to_idx[name]] = conc
    return vec

# --- Data Loading and Processing ---
def prepare_bioink_datasets(train_csv_path, test_csv_path, train_graph_dir, test_graph_dir, mat_to_idx, num_materials):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # 1. Parse all tabular features
    def extract_tabular_features(df):
        features = []
        for _, row in df.iterrows():
            components = parse_components(row['Components'])
            needle = parse_needle(row['Needle'])
            cells = parse_cells(row['Cells (e6/ml)'])
            
            feat = {
                'id': row['id'],
                'needle_diameter': needle['diameter'],
                'needle_cyl': 1.0 if needle['geometry'] == 'cylindrical' else 0.0,
                'needle_con': 1.0 if needle['geometry'] == 'conical' else 0.0,
                'cells': cells,
                'num_components': float(len(components)),
                'total_concentration': sum(c['concentration'] for c in components)
            }
            features.append(feat)
        return pd.DataFrame(features)

    train_tab = extract_tabular_features(train_df)
    test_tab = extract_tabular_features(test_df)
    
    # Scale numerical tabular columns
    num_cols = ['needle_diameter', 'needle_cyl', 'needle_con', 'cells', 'num_components', 'total_concentration']
    scaler = StandardScaler()
    train_scaled_num = scaler.fit_transform(train_tab[num_cols])
    test_scaled_num = scaler.transform(test_tab[num_cols])
    
    train_tab[num_cols] = train_scaled_num
    test_tab[num_cols] = test_scaled_num
    
    # Dictionary for fast lookup
    train_tab_dict = train_tab.set_index('id').to_dict('index')
    test_tab_dict = test_tab.set_index('id').to_dict('index')
    
    # 2. Build PyG Datasets
    def build_dataset(df, graph_dir, tab_dict, is_train=True):
        dataset = []
        for _, row in df.iterrows():
            gid = int(row['id'])
            
            # Load Raw matrices
            A = np.load(os.path.join(graph_dir, f'graph_{gid}_A.npy'))
            X = np.load(os.path.join(graph_dir, f'graph_{gid}_X.npy'))
            
            # Convert A to PyG Edge Index
            edge_indices = np.where(A > 0)
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            
            # Edge Features (concentrations of connected nodes)
            concs = X[:, -1]
            edge_features = []
            for k in range(edge_index.shape[1]):
                src = edge_index[0, k].item()
                dst = edge_index[1, k].item()
                c_src = concs[src]
                c_dst = concs[dst]
                feat = [c_src, c_dst, c_src * c_dst, abs(c_src - c_dst)]
                edge_features.append(feat)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            # Node features tensor
            x_tensor = torch.tensor(X, dtype=torch.float)
            
            # Global tabular features
            t_feat = tab_dict[gid]
            global_num = np.array([
                t_feat['needle_diameter'],
                t_feat['needle_cyl'],
                t_feat['needle_con'],
                t_feat['cells'],
                t_feat['num_components'],
                t_feat['total_concentration']
            ], dtype=np.float32)
            
            # Composition vector
            components = parse_components(row['Components'])
            comp_vec = get_composition_vector(components, mat_to_idx, num_materials)
            
            # Concatenate to form a 42-dimensional global attribute vector
            global_feat = np.concatenate([global_num, comp_vec])
            global_attr = torch.tensor(global_feat, dtype=torch.float).view(1, -1)
            
            y = None
            if is_train:
                y_data = np.array([row['pressure'], row['temperature'], row['speed']], dtype=np.float32)
                y = torch.tensor(y_data, dtype=torch.float).view(1, -1)
                
            data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.gid = gid
            data.global_attr = global_attr
            
            dataset.append(data)
            
        return dataset
        
    print("Building train dataset...")
    train_dataset = build_dataset(train_df, train_graph_dir, train_tab_dict, is_train=True)
    print("Building test dataset...")
    test_dataset = build_dataset(test_df, test_graph_dir, test_tab_dict, is_train=False)
    
    return train_dataset, test_dataset

# --- Custom Loss Function (Weighted NMAE) ---
PRESSURE_RANGE = 1496.0
TEMPERATURE_RANGE = 228.0
SPEED_RANGE = 90.0

PRESSURE_WEIGHT = 0.60
TEMPERATURE_WEIGHT = 0.25
SPEED_WEIGHT = 0.15

def weighted_nmae_loss(pred, target):
    mae_pressure = torch.mean(torch.abs(pred[:, 0] - target[:, 0]))
    mae_temp = torch.mean(torch.abs(pred[:, 1] - target[:, 1]))
    mae_speed = torch.mean(torch.abs(pred[:, 2] - target[:, 2]))
    
    nmae_pressure = mae_pressure / PRESSURE_RANGE
    nmae_temp = mae_temp / TEMPERATURE_RANGE
    nmae_speed = mae_speed / SPEED_RANGE
    
    loss = (PRESSURE_WEIGHT * nmae_pressure) + (TEMPERATURE_WEIGHT * nmae_temp) + (SPEED_WEIGHT * nmae_speed)
    return loss

def evaluate_metrics(preds, targets):
    # preds, targets: np.ndarray [N, 3]
    mae_press = np.mean(np.abs(preds[:, 0] - targets[:, 0]))
    mae_temp = np.mean(np.abs(preds[:, 1] - targets[:, 1]))
    mae_speed = np.mean(np.abs(preds[:, 2] - targets[:, 2]))
    
    nmae_press = mae_press / PRESSURE_RANGE
    nmae_temp = mae_temp / TEMPERATURE_RANGE
    nmae_speed = mae_speed / SPEED_RANGE
    
    combined_nmae = (PRESSURE_WEIGHT * nmae_press) + (TEMPERATURE_WEIGHT * nmae_temp) + (SPEED_WEIGHT * nmae_speed)
    
    return {
        'press_mae': mae_press,
        'temp_mae': mae_temp,
        'speed_mae': mae_speed,
        'press_nmae': nmae_press,
        'temp_nmae': nmae_temp,
        'speed_nmae': nmae_speed,
        'combined_nmae': combined_nmae
    }

def get_kfold_splits(df, n_splits=5, random_state=42):
    df['temp_regime'] = (df['temperature'] >= 50).astype(int)
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_idx, val_idx in sgkf.split(df, df['temp_regime'], groups=df['DOI']):
        splits.append((train_idx, val_idx))
    return splits

# --- Hybrid GNN Architecture ---
class HybridGNN(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, global_dim, out_channels=3):
        super(HybridGNN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=edge_dim, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, concat=False, edge_dim=edge_dim, dropout=0.2)
        
        pooled_dim = hidden_channels * 2
        
        self.regressor = nn.Sequential(
            nn.Linear(pooled_dim + global_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_channels)
        )
        
    def forward(self, x, edge_index, edge_attr, batch, global_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.elu(h)
        
        pooled_mean = global_mean_pool(h, batch)
        pooled_max = global_max_pool(h, batch)
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        
        x_all = torch.cat([pooled, global_attr], dim=1)
        out = self.regressor(x_all)
        return out

# --- Main Training & Inference ---
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data/public')
    
    train_csv_path = os.path.join(DATA_DIR, 'train.csv')
    test_csv_path = os.path.join(DATA_DIR, 'test_features.csv')
    train_graph_dir = os.path.join(DATA_DIR, 'train_graphs')
    test_graph_dir = os.path.join(DATA_DIR, 'test_graphs')
    vocab_path = os.path.join(DATA_DIR, 'node_vocabulary.txt')
    
    # Load Vocabulary
    mat_to_idx = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                idx_str, name = line.strip().split(',', 1)
                mat_to_idx[name] = int(idx_str)
    num_materials = len(mat_to_idx)
    print(f"Loaded material vocabulary of size: {num_materials}")
    
    # Load datasets
    train_dataset, test_dataset = prepare_bioink_datasets(
        train_csv_path, test_csv_path, train_graph_dir, test_graph_dir, mat_to_idx, num_materials
    )
    
    # Prepare DataFrame for splits
    df_train = pd.read_csv(train_csv_path)
    df_train['DOI'] = df_train['DOI'].fillna(df_train['id'].astype(str))
    
    # Create StratifiedGroupKFold splits
    splits = get_kfold_splits(df_train, n_splits=5, random_state=42)
    
    # Ensembling test predictions and validation tracking
    all_test_preds = np.zeros((len(test_dataset), 3), dtype=np.float32)
    
    val_scores_all_folds = []
    
    input_dim = train_dataset[0].x.shape[1]
    edge_dim = train_dataset[0].edge_attr.shape[1]
    global_dim = train_dataset[0].global_attr.shape[1]
    
    print(f"Model Specs -> Node input dim: {input_dim}, Edge input dim: {edge_dim}, Global input dim: {global_dim}")
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Training Fold {fold + 1} / 5 ---")
        
        # Split PyG datasets
        train_sub = [train_dataset[i] for i in train_idx]
        val_sub = [train_dataset[i] for i in val_idx]
        
        train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
        
        model = HybridGNN(
            in_channels=input_dim, 
            edge_dim=edge_dim, 
            hidden_channels=64, 
            global_dim=global_dim, 
            out_channels=3
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_nmae = float('inf')
        best_model_weights = None
        patience = 30
        patience_counter = 0
        
        for epoch in range(1, 301):
            # Train
            model.train()
            train_loss = 0
            for data in train_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_attr)
                loss = weighted_nmae_loss(out, data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.num_graphs
            avg_train_loss = train_loss / len(train_sub)
            
            # Eval
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(DEVICE)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_attr)
                    loss = weighted_nmae_loss(out, data.y)
                    val_loss += loss.item() * data.num_graphs
                    val_preds.append(out.cpu().numpy())
                    val_targets.append(data.y.cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_sub)
            scheduler.step(avg_val_loss)
            
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            scores = evaluate_metrics(val_preds, val_targets)
            
            if scores['combined_nmae'] < best_val_nmae:
                best_val_nmae = scores['combined_nmae']
                best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_scores = scores
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Train WNMAE: {avg_train_loss:.5f} | Val WNMAE: {scores['combined_nmae']:.5f} | Best Val WNMAE: {best_val_nmae:.5f}")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val WNMAE: {best_val_nmae:.5f}")
                break
                
        # Load best weights
        model.load_state_dict(best_model_weights)
        val_scores_all_folds.append(best_scores)
        
        # Predict on Test
        model.eval()
        fold_test_preds = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_attr)
                fold_test_preds.append(out.cpu().numpy())
        fold_test_preds = np.concatenate(fold_test_preds, axis=0)
        
        # Add to ensemble accumulator
        all_test_preds += fold_test_preds / 5.0
        
    # --- Print Aggregated Val Scores ---
    mean_val_nmae = np.mean([s['combined_nmae'] for s in val_scores_all_folds])
    mean_press_nmae = np.mean([s['press_nmae'] for s in val_scores_all_folds])
    mean_temp_nmae = np.mean([s['temp_nmae'] for s in val_scores_all_folds])
    mean_speed_nmae = np.mean([s['speed_nmae'] for s in val_scores_all_folds])
    
    print("\n" + "="*50)
    print("FINAL 5-FOLD VALIDATION PERFORMANCE SUMMARY:")
    print(f"  Combined NMAE:    {mean_val_nmae:.6f}")
    print(f"  Pressure NMAE:    {mean_press_nmae:.6f}")
    print(f"  Temperature NMAE: {mean_temp_nmae:.6f}")
    print(f"  Speed NMAE:       {mean_speed_nmae:.6f}")
    print("="*50)
    
    # Save test predictions
    test_df = pd.read_csv(test_csv_path)
    pred_df = pd.DataFrame(all_test_preds, columns=['pressure', 'temperature', 'speed'])
    pred_df.insert(0, 'id', test_df['id'])
    
    # Ensure directory run_gemini exists
    os.makedirs(os.path.join(BASE_DIR, 'run_gemini'), exist_ok=True)
    
    predictions_path = os.path.join(BASE_DIR, 'run_gemini', 'predictions.csv')
    pred_df.to_csv(predictions_path, index=False)
    print(f"\nSaved test predictions to: {predictions_path}")
    
    # Save run_summary.csv
    summary_df = pd.DataFrame([{
        'val_f1': mean_val_nmae,
        'val_acc': np.mean([s['press_mae'] for s in val_scores_all_folds]), # we can write MAE here
        'threshold': 0.0,
        'epochs': 300,
        'splits': '5-fold DOI stratified',
        'val_nmae': mean_val_nmae,
        'val_pressure_nmae': mean_press_nmae,
        'val_temp_nmae': mean_temp_nmae,
        'val_speed_nmae': mean_speed_nmae
    }])
    summary_path = os.path.join(BASE_DIR, 'run_gemini', 'run_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved run summary to: {summary_path}")

if __name__ == "__main__":
    main()
