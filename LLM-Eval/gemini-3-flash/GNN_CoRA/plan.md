# Plan (Gemini CLI, zero-shot, CPU)

`<plan>` produced under the frozen prompt template instantiated for **GNN_CoRA** — node classification on the Cora citation network with noisy features.

1. **Task framing.** 2,708 nodes, 5,429 undirected edges. Node features are 1,433-dim bag-of-words with Gaussian noise (σ = 0.4). 7 classes. Small training set (140 nodes), 500 validation nodes, 1,000 unlabelled test nodes.
2. **Node features.** Use the provided 1,433 features from `x.csv`. These represent normalized word frequencies across scientific publications.
3. **Graph structure.** Represent the citation network using PyTorch Geometric's `Data` object. Symmetric adjacency normalization is handled automatically by the `GCNConv` layers.
4. **Architecture.** 2-layer Graph Convolutional Network (GCN).
   - Layer 1: `GCNConv(1433, 64)` followed by ReLU and Dropout (p=0.5).
   - Layer 2: `GCNConv(64, 7)` to produce logits for 7 classes.
5. **Loss.** Negative Log Likelihood (`NLLLoss`) applied to log-softmax outputs, calculated over the training nodes.
6. **Training protocol.** Adam optimizer with learning rate 0.01 and weight decay 5e-4. Train for 200 epochs, performing full-batch gradient descent.
7. **Validation strategy.** Evaluate validation accuracy every epoch on the 500 labeled validation nodes. Implement model checkpointing to save the state with the highest validation accuracy.
8. **Inference.** Load the best-performing model state. Execute a forward pass on the full graph. Extract predictions (argmax) for the 1,000 test IDs provided in `data/test_ID.csv`.
9. **Submission file.** Generate `predictions.csv` and `submission.csv` at the repository root. Ensure the format adheres to the required `id,target` columns with integer predictions.
10. **Reproducibility.** Fixed seeds for `torch`, `numpy`, and `random` (Seed=42). CPU-only execution targeted to finish in under 1 minute.
