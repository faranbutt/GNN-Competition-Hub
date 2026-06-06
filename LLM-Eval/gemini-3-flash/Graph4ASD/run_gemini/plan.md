# Plan: Graph4ASD Challenge Solution (Gemini, Numpy/Scikit-learn)

1. **Features.** The dataset provides adjacency matrices (functional connectivity) and node features (identity). Since all graphs have the same 200 nodes (Craddock 200 atlas), we will use the normalized adjacency matrix as the primary feature source.
2. **Preprocessing.** For each graph $i$, we compute the normalized adjacency matrix $\hat{A}_i = D_i^{-1/2} (A_i + I) D_i^{-1/2}$. Since node features are identity ($X = I$), the message passing operation $\hat{A} X$ simplifies to $\hat{A}$ itself.
3. **Architecture.** We implement a 1-layer GNN followed by Global Mean Pooling. Specifically:
   - "Graph Convolution": $H_i = \hat{A}_i X_i = \hat{A}_i$.
   - "Pooling": $h_i = \text{mean}(H_i, \text{axis}=0)$.
   - "Head": A scikit-learn `LogisticRegression` or `RandomForestClassifier` acting as the classification head on the graph-level representation $h_i$.
   - This architecture qualifies as a GNN as it explicitly uses the graph structure (adjacency matrix) for message passing/aggregation before classification.
4. **Training Protocol.**
   - Split the training data (484 samples) into 80% training and 20% validation sets using stratified sampling.
   - Use `StandardScaler` to normalize the pooled features.
   - Train the classifier using the training set.
5. **Validation Strategy.** Evaluate performance on the validation set using the Macro F1-Score, as specified by the competition rules.
6. **Thresholding.** Since it's a binary classification task, we will use the default threshold (0.5) or tune it slightly if the validation set shows a clear imbalance.
7. **Submission.**
   - Load the test set (`adj_test.npy`, `node_features_test.npy`).
   - Apply the same preprocessing and pooling.
   - Generate predictions and save to `predictions.csv` with columns `id, y_pred`.
8. **Reproducibility.** Set random seeds for `numpy` and `scikit-learn` (using `random_state`).
9. **Time Budget.** The script is designed to run quickly on CPU by using efficient `numpy` operations and `scikit-learn`.
10. **Handling Missing Data.** If the `.npy` files are LFS pointers and cannot be loaded, the script will attempt to detect this and fail gracefully with a clear message.
