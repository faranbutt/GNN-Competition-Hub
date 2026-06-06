<plan>
- Features: Use the four provided morphological features: x, y, width, and height. 
- Model Architecture: A 2-layer Graph Convolutional Network (GCN).
  - Layer 1: GCN layer with hidden dimension 64, ReLU activation, and Dropout (0.5).
  - Layer 2: GCN layer with output dimension 4 (number of cell types).
- Training Protocol:
  - Loss: CrossEntropyLoss with class weights calculated from the training labels to handle imbalance.
  - Optimizer: Adam with a learning rate of 0.01 and weight decay of 5e-4.
  - Epochs: 200 epochs.
- Validation Strategy: Randomly split 20% of the training nodes into a local validation set to monitor Macro F1 and Accuracy.
- Threshold/Decoding: Use argmax over the model's output logits for class prediction.
- Submission: Save predictions for the test nodes to `run_gemini-3-flash/predictions.csv` with columns `id` and `y_pred`.
</plan>
