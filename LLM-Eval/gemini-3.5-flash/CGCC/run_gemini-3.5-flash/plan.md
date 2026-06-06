# City Graph Class Challenge (CGCC) Plan

Our solution is designed to tackle the urban street network layout classification task by leveraging a combination of local graph convolution and global network topologies.

## 1. Feature Engineering

We extract two levels of features from each city graph `G` (which is a serialized NetworkX `MultiGraph`):

### Node Features
For each node $i \in V$, we construct a 9-dimensional feature vector $X_i \in \mathbb{R}^9$:
1. **Centered & Scaled coordinates $(x, y)$** (2 dimensions):
   $$x_i' = \frac{x_i - \mu_x}{\sigma}, \quad y_i' = \frac{y_i - \mu_y}{\sigma}$$
   where $\sigma = \sqrt{\text{Var}(x) + \text{Var}(y)} + 1\text{e-}6$ centers the graph and preserves scale.
2. **Normalized Node Degree** (1 dimension):
   $$d_i' = \frac{d_i - \mu_d}{\sigma_d + 1\text{e-}6}$$
3. **One-Hot Encoded Street Count** (5 dimensions):
   OSM node attribute `street_count` represents physical road intersections. We map this into 5 bins: 1, 2, 3, 4, and $\ge 5$.
4. **Local Clustering Coefficient** (1 dimension):
   $$c_i = \text{nx.clustering}(G)[i]$$
   Measures local interconnectivity and triangle density, distinguishing organic winding streets from rigid grid structures.

### Global Graph-Level Features
We engineer a 12-dimensional topological vector $g \in \mathbb{R}^{12}$ to capture the macro characteristics of each network:
1. `num_nodes`: Total intersections (scaled by $0.01$).
2. `num_edges`: Total street segments (scaled by $0.01$).
3. `edge_node_ratio`: Average streets per intersection.
4. `deg_mean`: Mean node degree.
5. `deg_std`: Standard deviation of node degrees.
6. `sc_mean`: Mean street count.
7. `sc_std`: Standard deviation of street counts.
8. `pct_sc3`: Proportion of 3-way intersections (classic historic organic cities).
9. `pct_sc4`: Proportion of 4-way intersections (strongly indicative of grid cities).
10. `len_mean`: Average segment length (scaled by $0.01$).
11. `len_cv`: Coefficient of variation of segment lengths (captures regularity of block sizes).
12. `entropy_mod`: Entropy of edge orientations (bearings) modulo $90^\circ$ (highly discriminative: grids align to tight orthogonal bins, while organic networks have higher orientation entropy).

---

## 2. Model Architecture

The neural network (`GNNFusionModel`) is composed of three parts:

### Deep Residual GCN
- Input: Node features $X \in \mathbb{R}^{N \times 9}$ and normalized adjacency matrix $A_{norm} \in \mathbb{R}^{N \times N}$.
- Process: Three successive Graph Convolutional layers with skip connections:
  $$H^{(l+1)} = \text{ReLU}\left(\text{BatchNorm}\left(A_{norm} H^{(l)} W^{(l)}\right)\right) + \text{Shortcut}(H^{(l)})$$
- Normalization: $A_{norm} = D^{-1/2}(A + I)D^{-1/2}$.
- Hidden dimension: 64. Dropout: 0.2.

### Multi-scale Pooling (Readout)
- Summarizes the node embeddings into a graph embedding:
  $$h_{graph} = \text{Concat}\left(\text{MeanPool}(H^{(3)}), \text{MaxPool}(H^{(3)}), \text{SumPool}(H^{(3)})\right)$$
  - MeanPool: Captures overall structural average.
  - MaxPool: Identifies the most prominent local features.
  - SumPool: Retains scale and network size information.
- Dimension: $64 \times 3 = 192$.

### Global Feature MLP & Fusion
- Projects global features $g \in \mathbb{R}^{12}$ to a dense representation:
  $$h_{global} = \text{BatchNorm}\left(\text{ReLU}\left(\text{Linear}_{12 \to 32}(g)\right)\right)$$
- Fuses local GNN structure and global macro topologies:
  $$h_{fused} = \text{Concat}(h_{graph}, h_{global}) \in \mathbb{R}^{224}$$
- Classification MLP:
  $$\text{Logits} = \text{Linear}_{64 \to 3}\left(\text{ReLU}\left(\text{Dropout}\left(\text{Linear}_{224 \to 64}(h_{fused})\right)\right)\right)$$

---

## 3. Training & Validation Protocol

- **Validation Scheme**: 5-Fold Stratified Cross-Validation (highly stable for small datasets like CGCC's 84 graphs).
- **Optimizer**: Adam with learning rate 0.005 and weight decay 1e-4.
- **Loss**: Weighted Cross-Entropy Loss (rebalances the training labels based on the inverse class counts).
- **LR Schedule**: Cosine Annealing learning rate schedule.
- **Ensembling**: We save the best state dictionary for each of the 5 folds. The final predictions for the test set are computed by averaging the soft probabilities (softmax of logits) across all 5 models.
