# Plan: NetLinkArena Link Prediction

<plan>
1. **Data Loading & Preprocessing**:
   - Load `node_features.csv` as the initial node feature matrix (3327 nodes, 2742 features).
   - Normalize node features using row-wise L2 normalization to handle sparsity and scale differences.
   - Load `train_edges.csv`, `val_edges.csv`, and `test_nodes.csv`.

2. **Graph Construction**:
   - Construct an undirected graph for message passing using only the positive edges (label=1) from `train_edges.csv`.
   - Convert these edges to a `torch_geometric` compatible `edge_index`.

3. **Model Architecture**:
   - **Encoder**: A 2-layer GraphSAGE (SAGEConv) model.
     - Layer 1: SAGEConv(2742 -> 128) + ReLU + Dropout(0.5).
     - Layer 2: SAGEConv(128 -> 64) + ReLU.
   - **Decoder**: A simple dot product between the embeddings of the source and target nodes to compute the link score.

4. **Training Protocol**:
   - **Loss Function**: `BCEWithLogitsLoss` applied to the scores of training edge pairs (both positive and negative).
   - **Optimizer**: Adam with a learning rate of 0.01 and weight decay of 5e-4.
   - **Epochs**: 200 epochs with early stopping based on Validation ROC-AUC.

5. **Validation Strategy**:
   - Evaluate the model on `val_edges.csv` after each epoch.
   - Calculate ROC-AUC score to monitor performance and save the best model state.

6. **Inference & Submission**:
   - Use the best model to generate embeddings for all nodes.
   - Compute link scores for test pairs in `test_nodes.csv`.
   - Apply the sigmoid function to convert logits to probabilities.
   - Save the results as `predictions.csv` with `id` and `y_pred` columns.
</plan>
