# Plan - Gemini-3-flash

## Task Overview
- **Goal**: Classify city street network graphs into three classes: `organic` (0), `grid` (1), and `hybrid` (2).
- **Metric**: Macro-F1 score.
- **Data**: 120 cities (70% train, 30% test). Unbalanced classes.

## Proposed Strategy
1. **Features**:
   - **Node Features**: 
     - Centered and scaled `x`, `y` coordinates.
     - Normalized node `degree`.
     - (Optional) Local clustering coefficient if computationally feasible.
   - **Graph Features**: 
     - Number of nodes and edges (normalized).
     - Average degree.
2. **Model Architecture**:
   - 3-layer Graph Convolutional Network (GCN).
   - Each layer: `Linear` -> `BatchNorm` -> `ReLU` -> `Dropout`.
   - Global Pooling: Concatenate `GlobalAveragePooling` and `GlobalMaxPooling`.
   - Classification Head: 2-layer MLP with `ReLU`.
3. **Training Protocol**:
   - Stratified 5-Fold Cross-Validation to ensure robustness and handle class imbalance.
   - Optimizer: `Adam` with weight decay.
   - Loss: `CrossEntropyLoss` with class weights.
   - Epochs: 200-300 with Early Stopping based on validation Macro-F1.
4. **Validation Strategy**:
   - Use stratified splits to maintain class distribution.
   - Track both Accuracy and Macro-F1.
5. **Ensemble**:
   - Average predictions (logits or probabilities) from the 5 fold models for the final test set prediction.
6. **Submission**:
   - Generate `predictions.csv` with `filename` and `prediction` columns.
   - Filenames will be sorted to match expected order if necessary.

## Implementation Details
- Implementation will use `torch` for the GNN (manual sparse matrix multiplication for GCN layers to stay within `requirements.txt` constraints).
- `networkx` will be used for graph loading and initial feature extraction.
- The script will be self-contained and runnable on CPU.
