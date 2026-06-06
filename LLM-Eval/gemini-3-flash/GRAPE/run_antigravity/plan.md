<plan>
1. **Features**: Use the 4 provided node features: `x`, `y` (normalized coordinates), `width` (vessel width), and `type` ( junction vs endpoint encoded as binary). Additionally, extract 5 graph-level topological features: normalized node count, normalized edge count, average node degree, junction ratio, and average vessel width.
2. **Model Architecture**: A 3-layer Graph Attention Network (GAT) model with multi-head attention (4 heads, dropout=0.3) and batch normalization. For pooling, use a combination of global mean, max, and add pooling to capture different statistical moments of the node feature distributions. Concatenate these with the graph-level topological features before a final 2-layer MLP classifier.
3. **Training Protocol**: 
    - Optimizer: AdamW with a learning rate of 0.005 and weight decay of 1e-4.
    - Learning Rate Scheduler: CosineAnnealingLR for smooth decay.
    - Loss: CrossEntropyLoss with normalized class weights to counteract the ~69%/31% class imbalance.
    - Early stopping with patience=40 based on validation Macro F1 score.
4. **Validation Strategy**: Use a stratified 80/20 train/val split based on the ground truth healthy vs DR labels to ensure stable metric estimation.
5. **Threshold/Decoding**: Predictions are binary, so the argmax over logits is used directly (which corresponds to threshold=0.5).
6. **Submission**: Save final test predictions to `predictions.csv` with columns `graph_id` and `label`.
</plan>
