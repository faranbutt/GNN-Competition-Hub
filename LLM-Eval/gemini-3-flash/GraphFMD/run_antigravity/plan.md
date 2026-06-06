# Plan: Graph-based Financial Misconduct Detection (GraphFMD)

## Goal
Train a Graph Neural Network (GNN) to classify Bitcoin transactions as illicit or licit using the provided temporal graph data.

## 1. Features and Data Processing
- **Features**: Utilize all 165 node features provided in `train_nodes.csv` and `test_nodes.csv`.
- **Normalization**: Apply `StandardScaler` to normalize features across the combined train and test sets.
- **Mapping**: Map transaction IDs to continuous integer indices for graph representation.
- **Edges**: Construct the graph's `edge_index` from `edgelist.csv`, preserving the directed nature of transaction flows.

## 2. Model Architecture
- **Type**: Graph Convolutional Network (GCN).
- **Layers**: 2-layer GCN.
  - **Layer 1**: `GCNConv` (165 -> 64) with ReLU activation.
  - **Layer 2**: `GCNConv` (64 -> 2) for binary classification.
- **Regularization**: Dropout (p=0.3) after the first layer to prevent overfitting.

## 3. Training Protocol
- **Optimizer**: Adam with a learning rate of 0.01 and weight decay of 5e-4.
- **Loss Function**: Cross-Entropy Loss.
- **Class Imbalance**: Compute and apply class weights (inverse frequency) in the loss function to handle the sparsity of illicit transactions.
- **Epochs**: 100 epochs on CPU.

## 4. Validation Strategy
- **Split**: 80/20 stratified split on the labeled training nodes to ensure both classes are represented in training and validation sets.
- **Metric**: Macro-F1 score (as per competition rules).
- **Model Selection**: Save and use the model state with the highest validation Macro-F1.

## 5. Threshold and Decoding
- **Threshold**: Standard 0.5 threshold on softmax probabilities (equivalent to `argmax` on logits).
- **Output Mapping**: Map class 1 back to label `1` (Illicit) and class 0 to label `2` (Licit).

## 6. Submission
- Generate `predictions.csv` containing `id` and `y_pred`.
- Produce `run_summary.csv` with validation metrics and training parameters.
