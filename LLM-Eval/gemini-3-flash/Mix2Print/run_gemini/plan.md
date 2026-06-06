# GNN Architecture and Training Protocol

This document details the architecture, features, and training protocol designed for the **Mix2Print GNN Challenge**.

## 1. Feature Engineering
We use a two-pronged feature representation strategy: Graph-level features and Node/Edge features.

### A. Graph Representation
- **Nodes**: Biomaterials present in the formulation (derived from `Components`).
- **Node Features (37 dimensions)**: 36-dimensional one-hot encoding of biomaterial identities + 1-dimensional normalized concentration.
- **Edges**: Fully-connected clique (representing chemical mixtures where all components interact).
- **Edge Features (4 dimensions)**: For an edge between node $i$ (concentration $C_i$) and node $j$ (concentration $C_j$):
  $$E_{ij} = [C_i, C_j, C_i \times C_j, |C_i - C_j|]$$
  This captures absolute concentrations, concentration products (interaction strength), and concentration gradients (diffusion/interaction dynamics).

### B. Global Tabular Features
These are injected directly into the regressor head to provide context that cannot be represented simply by composition graph structure:
1. **Needle Diameter (Numerical)**: Extracted from `Needle` (converts Gauge to µm, defaults to mean diameter 400 µm).
2. **Needle Geometry (One-Hot)**: `cylindrical` and `conical` flags.
3. **Cell Concentration (Numerical)**: Extracted from `Cells (e6/ml)`.
4. **Graph Complexity**:
   - `num_components`: Number of nodes in the graph.
   - `total_concentration`: Sum of concentrations.
5. **Raw Material Composition Vector (36 dimensions)**: Sum of concentration of each of the 36 biomaterials in the mixture.

---

## 2. GNN Model Architecture
We build a hybrid model combining a message-passing GNN with a global MLP regressor:

```
Nodes (X)  ───►  GATv2 Layer 1 (64 hidden)  ───►  GATv2 Layer 2 (64 hidden)  ───►  Global Pooling
Edges (E)  ───▲                              ───▲                                          │
                                                                                           ▼
Tabular Features (Needle, Cells, Composition sums, etc.) ───────────────────────►  Concatenation
                                                                                           │
                                                                                           ▼
Output Targets ◄─── Linear ◄─── ReLU ◄─── Dropout (0.3) ◄─── Linear ◄─── MLP Regressor Head
```

### Details:
- **GNN Layers**: Two layers of `GATv2Conv` with multi-head attention (4 heads, hidden size 64) and edge feature support (`edge_dim=4`).
- **Pooling**: Concatenate both `global_mean_pool` and `global_max_pool` (total 128 hidden dim) to capture average composition characteristics and peak component contributions.
- **Aggregation**: Concatenate the 128-dim graph embedding with the 42-dim global tabular features (total 170-dim input to the regressor).
- **Regressor Head**: An MLP with layer sizes: `[170 -> 128 -> 64 -> 3]`.

---

## 3. Training Protocol & Validation Strategy

### A. Validation Split
We perform a **DOI-based Stratified Group 5-Fold Cross-Validation**:
- **Group Grouping**: Grouped by paper `DOI` so that all formulations from a single publication are strictly in either train or validation set. This prevents data leakage (since papers have distinct experimental protocols).
- **Stratification**: Stratify the group splits based on the temperature regime (hydrogel: $< 50^\circ\text{C}$ vs thermoplastic: $\ge 50^\circ\text{C}$) to maintain target distributions across folds.

### B. Loss Function
Directly optimize the **Weighted Normalized Mean Absolute Error (WNMAE)**:
$$\mathcal{L} = 0.60 \times \frac{\text{MAE}_{\text{pressure}}}{1496.0} + 0.25 \times \frac{\text{MAE}_{\text{temp}}}{228.0} + 0.15 \times \frac{\text{MAE}_{\text{speed}}}{90.0}$$
This aligns the training objective exactly with the leaderboard evaluation metric.

### C. Optimization Details
- **Optimizer**: AdamW with learning rate $5 \times 10^{-3}$ and weight decay $1 \times 10^{-4}$.
- **Scheduler**: `ReduceLROnPlateau` with patience 10, factor 0.5.
- **Epochs**: Up to 300 epochs per fold with Early Stopping (patience 30 epochs on validation WNMAE).
- **Ensemble Prediction**: The test set predictions are the average of predictions from models trained across all 5 folds, significantly boosting generalization and reducing variance.
