# GNN Implementation Plan - Glimps-GNN

## Goal
Train an inductive GNN to predict preeclampsia (binary classification) by transferring knowledge from cfRNA (maternal plasma) to placenta samples.

## Proposed Changes

### Features
- **Node Features**: Gene expression values (6650 genes) from `train.csv` and `test.csv`.
- **Normalization**: Standard scaling of features across all nodes (inductive-safe: fit on train, transform all).
- **Dimensionality Reduction**: Since features (6650) >> samples (320), a linear projection layer will be used as the first step in the GNN model to map features to a lower-dimensional latent space.

### Model Architecture
- **Type**: Inductive GraphSAGE (SAGEConv layers) to handle the inductive transfer.
- **Layers**: 
    1. Linear Projection (6650 -> 256)
    2. SAGEConv (256 -> 128) + BatchNorm + ReLU + Dropout
    3. SAGEConv (128 -> 64) + BatchNorm + ReLU + Dropout
    4. MLP Head (64 -> 1) with Sigmoid activation.

### Training Protocol
- **Loss Function**: Binary Cross Entropy with Logits (BCEWithLogitsLoss).
- **Optimizer**: Adam with weight decay (L2 regularization) to prevent overfitting.
- **Learning Rate**: 0.001 with a scheduler (ReduceLROnPlateau).
- **Epochs**: 200 with Early Stopping based on validation F1.
- **Batching**: Full-batch training since the graph size (320 nodes) is small enough for CPU memory.

### Validation Strategy
- **Stratified 5-Fold Cross-Validation** on the `cfRNA` (training) nodes.
- Final predictions will be an ensemble (average) of the 5 models.

### Threshold/Decoding
- Threshold optimization: Find the threshold that maximizes F1 score on each validation fold and average these thresholds for final test set inference.

### Submission File
- The output `predictions.csv` will contain:
    - `id`: The `node_id` for placenta samples (from `test_nodes.csv`).
    - `y_pred`: Binary prediction (0 or 1) based on the optimized threshold.

## Execution Details
- All training and inference will be done on CPU.
- Random seeds will be fixed (torch, numpy, random) for reproducibility.
