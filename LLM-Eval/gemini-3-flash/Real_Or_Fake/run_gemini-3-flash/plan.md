# Plan: Real Or Fake?! GNN Challenge

<plan>
1. **Feature Engineering**:
   - Concatenate three feature sets for each node: BERT (768-dim), spaCy (300-dim), and User Profile features (10-dim).
   - Total input feature dimension: 1078.
   - Use `scipy.sparse.csr_matrix` to load sparse features from `.npz` files and convert to dense tensors for the GNN.

2. **Model Architecture**:
   - **Encoder**: 3-layer GraphSAGE (SAGEConv) network. GraphSAGE is robust for inductive learning on graphs.
     - Layers: SAGEConv(1078->128), SAGEConv(128->128), SAGEConv(128->128).
     - Activation: ReLU.
     - Dropout: 0.3 for regularization.
   - **Pooling**: Combine `global_mean_pool` and `global_max_pool` of node embeddings to capture both average and extreme features of the propagation graph.
   - **Root Node Context**: Explicitly extract the embedding of the root node (news article) and pass it through a linear layer.
   - **Classifier**: Concatenate the pooled graph representation (256-dim) and the root node representation (128-dim). Final MLP: Linear(384->1) + Sigmoid.

3. **Training Protocol**:
   - **Loss**: Binary Cross Entropy Loss (`BCELoss`).
   - **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-4.
   - **Batch Size**: 64 graphs per batch.
   - **Epochs**: 50 epochs with early stopping based on validation accuracy.
   - **Reproducibility**: Set seeds for `numpy`, `torch`, and `random`.

4. **Validation Strategy**:
   - Use the provided `val_idx.npy` and `val_labels.csv` to monitor performance during training.
   - Save the model state that achieves the highest validation accuracy.

5. **Inference & Submission**:
   - Load the best model and run inference on the test set (`test_idx.npy`).
   - Use a threshold of 0.5 to convert probabilities to hard labels (0 or 1).
   - Write the results to `predictions.csv` with columns `id` and `y_pred`.
</plan>
