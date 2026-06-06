# OASIS-3 Brain Age Prediction: GNN Plan

This plan documents the specific details of the GNN architecture, feature engineering, and training protocol utilized to solve the OASIS-3 chronological age prediction task.

## 1. Feature Engineering & Preprocessing
- **Node Features (68 regions, 2 features/region)**:
  - We extract Gray Matter Volume and Cortical Thickness for each of the 68 regions in the Desikan-Killiany atlas.
  - Features are standardized region-by-region across patients using `StandardScaler` fitted exclusively on the training set to prevent leakage.
  - The final feature vector for each graph session is reshaped into `(68, 2)`.
- **Edge Connectivity & Weights (dMRI streamline counts)**:
  - Adjacency matrices are loaded from `data/public/adjacency_matrices/`.
  - Streamline counts are log-transformed via `log1p` to compress their wide range and then normalized to `[0, 1]` using the training maximum.
  - We preserve the natural self-loops (diagonal entries) as they represent meaningful intra-regional connectivity. GCNConv is configured with `add_self_loops=False` to prevent dilution.

## 2. GNN Model Architecture
- **Type**: Graph Convolutional Network (GCN).
- **Conv Layers**: 3 layers of `GCNConv` (`2 -> 64 -> 64 -> 64`) with ReLU activation and dropout (`p=0.2`).
- **Graph Aggregation**: Global mean pooling concatenated with global max pooling (`global_mean_pool` + `global_max_pool`) to yield a robust 128-dimensional graph embedding.
- **MLP Regressor**:
  - `fc1`: `Linear(128, 64)` + ReLU + Dropout
  - `fc2`: `Linear(64, 32)` + ReLU
  - `fc3`: `Linear(32, 1)` (predicts chronological age)

## 3. Training & Hyperparameters
- **Loss**: L1 Loss (MAE) to directly align training with the primary competition metric.
- **Optimizer**: Adam with learning rate `0.005`, weight decay `1e-4`.
- **Batch Size**: 16
- **Epochs**: 120
- **Validation Strategy**: We monitor validation MAE at every epoch and save the best model checkpoint to make predictions on the test set.
