# PROVEN-GNN Implementation Plan

## Goal
The goal is to train a Graph Neural Network (GNN) to classify code functions as vulnerable or non-vulnerable using Code Property Graphs (CPG).

## Proposed Changes
### Features
- **Node Features**: 527-dimensional vectors (15 for node type, 512 for code embeddings).
- **Edge Features**: 12 types of relationships (AST, CFG, PDG, etc.).

### Architecture
- **Model**: A GNN with Graph Attention Network (GAT) layers. GAT is chosen for its ability to weigh neighbors differently and handle edge attributes.
- **Layers**: 2-3 GAT layers with hidden dimension 128.
- **Pooling**: Combination of Global Mean Pooling and Global Max Pooling to capture diverse graph properties.
- **Output**: Fully connected layers followed by a Softmax/LogSoftmax for binary classification.

### Training Protocol
- **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-4.
- **Loss Function**: CrossEntropyLoss.
- **Epochs**: 100 epochs with early stopping based on validation F1.
- **Reproducibility**: Set seeds for torch, numpy, and random.

### Validation Strategy
- **Split**: 80/20 stratified split of the training data.
- **Metric**: Macro F1-Score (primary metric for the competition).

### Submission
- Generate `predictions.csv` containing `id` and `y_pred` for the test set.

## Verification Plan
- Monitor training and validation loss/F1.
- Ensure `predictions.csv` matches the required format.
