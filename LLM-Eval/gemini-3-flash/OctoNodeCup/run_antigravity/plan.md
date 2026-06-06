<plan>
1. **Features**: Load 42-dimensional node features from `node_features.npy` and the sparse adjacency matrix from `adjacency_matrix.npz`. 
2. **Model Architecture**: A 2-layer GraphSAGE (SAGEConv) model. SAGE is chosen because it handles massive graph density (~289K edges) gracefully, resists overfitting on social networks, and allows flexible feature aggregation. We'll use 128 hidden channels, Batch Normalization, and Dropout (0.5).
3. **Training Protocol**: 
    - Optimizer: Adam with a learning rate of 0.01 and weight decay of 5e-4.
    - Loss: BCEWithLogitsLoss with `pos_weight` to address imbalance.
    - Early stopping with patience=30 on Validation Macro F1.
4. **Validation Strategy**: Use a stratified 80/20 train/val split based on the `ml_target` labels in `train_target.csv`.
5. **Threshold/Decoding**: Macro F1 optimizes heavily around balanced thresholds; a threshold search will be run in [0.3, 0.7] on validation probabilities to secure the best validation score.
6. **Submission**: Save final test predictions to `predictions.csv` with exactly 3 columns: `id`, `name`, `ml_target` matching the exact ordering of `test_target_without_labels.csv`.
</plan>
