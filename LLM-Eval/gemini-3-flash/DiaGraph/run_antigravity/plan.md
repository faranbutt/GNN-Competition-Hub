<plan>
1. **Features**: Load node features from `nodes.csv`. Handle missing values in `HbA1c_level` by filling them with 0 (since a `HbA1c_missing` indicator is already provided). All features except `id` will be normalized if necessary, though they appear pre-normalized in the head.
2. **Model Architecture**: A 2-layer GraphSAGE (SAGEConv) model. GraphSAGE is efficient for large-scale graphs (~96k nodes) and robust to noise. We will use a hidden dimension of 128, ReLU activations, and dropout (0.5) to prevent overfitting.
3. **Training Protocol**: 
    - Optimizer: Adam with a learning rate of 0.01 and weight decay of 5e-4.
    - Loss Function: `BCEWithLogitsLoss` with `pos_weight` to address class imbalance.
    - Training will be performed on nodes specified in `train.csv` and validated on `val.csv`.
    - Stop training based on the best validation Macro F1 score (evaluated with threshold 0.5).
4. **Validation Strategy**: Use the pre-defined `val.csv` split to monitor performance and ensure robustness.
5. **Threshold/Decoding**: As per competition rules, a threshold of 0.5 will be used for final classification, but the submission requires probabilities.
6. **Submission**: Generate probabilities for all IDs in `test_nodes.csv` and save them in the required CSV format (`id,y_pred`).
</plan>
