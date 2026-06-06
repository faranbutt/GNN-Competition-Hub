<plan>
1. **Features**: Use the provided 8-dimensional node features (`nf_0` to `nf_7`) and the edge list. Edge features (`ef_0`) will be included if they represent bond types.
2. **Model Architecture**: A Graph Attention Network (GAT) with 3 layers, hidden dimension of 64, and 4 heads. This will be followed by global mean pooling and global max pooling (concatenated) to capture graph-level features, then a 2-layer MLP for binary classification.
3. **Training Protocol**: 
    - Optimizer: Adam with learning rate 1e-3.
    - Loss: BCEWithLogitsLoss with a `pos_weight` calculated from the training set to handle the 10% class imbalance.
    - Scheduler: StepLR or CosineAnnealing to decay the learning rate.
    - Duration: Train for 100 epochs or until early stopping (patience=20) triggers based on validation Macro F1.
4. **Validation Strategy**: Use a stratified 80/20 train-validation split (based on labels) to ensure representative performance measurement.
5. **Threshold/Decoding**: Since the metric is Macro F1, the optimal classification threshold will be determined by searching over values [0.1, 0.9] on the validation set.
6. **Submission**: Predictions for the test set will be generated using the optimized threshold and saved as `predictions.csv` with columns `id` and `target`.
</plan>
