<plan>
- Features: Use all 22 acoustic voice measurement features provided in the node features.
- Model Architecture: A Graph Attention Network (GAT) with 2 GAT layers. 
  - Layer 1: GATConv with 8 attention heads, hidden dimension 32, and ELU activation.
  - Layer 2: GATConv with 1 attention head, output dimension 2 (binary classes).
  - Use Dropout (0.6) for regularization to prevent overfitting on the small dataset.
- Training Protocol:
  - Loss: CrossEntropyLoss with class weights to handle imbalance (approx 1:3 Healthy to Parkinson).
  - Optimizer: Adam with learning rate 0.005 and weight decay 5e-4.
  - Epochs: Up to 500 with early stopping (patience 50) based on Validation Macro F1.
- Validation Strategy: Use the pre-defined `val_mask` in the dataset to monitor Macro F1-Score.
- Threshold/Decoding: Standard argmax over logits for binary classification.
- Submission: Write a CSV file with columns `node_id` and `prediction` for the 39 test nodes.
</plan>
