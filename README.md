# Multimodal Deep Canonical Correlation Analysis with Deep Embedded Clustering (CCA-DEC)

## 📌 Project Overview
This project implements a deep learning framework to jointly analyze **resting-state fMRI brain connectivity** and **behavioral/cognitive measures** from the ABCD Study. It utilizes **Deep Canonical Correlation Analysis (DCCA)** to learn correlated latent representations across the two modalities, followed by **Deep Embedded Clustering (DEC)** to identify distinct neurological subgroups without labeled data.

Two behavioral datasets are supported:
- **CBCL** (Child Behavior Checklist) — 11 psychopathology subscales (anxiety/depression, attention, aggression, internalizing, externalizing, etc.)
- **NIH Toolbox** — 10 age-corrected cognitive composite scores

The model discovers clusters that map onto clinically meaningful profiles (e.g., low-severity, subclinical, clinical), validated against a linear CCA baseline and assessed via permutation tests with FDR correction.

## 🛠️ Technical Stack
* **Framework:** PyTorch
* **Language:** Python
* **Libraries:** NumPy, Pandas, Scikit-learn, SciPy, Statsmodels, Matplotlib, UMAP, Nilearn
* **Models:** Multimodal Autoencoders, Deep CCA, Deep Embedded Clustering (DEC)

## 🚀 Key Features
* **Multimodal Fusion:** Jointly encodes high-dimensional brain connectivity matrices (RSFC, ~5,952 subjects) and behavioral/cognitive measures into shared latent representations.
* **Unsupervised Clustering:** Applies DEC on the concatenated latent space with K-Means initialization to discover interpretable subject subgroups.
* **Custom Loss Functions:** Combined loss of reconstruction (MSE), Canonical Correlation (via Cholesky-whitened SVD), and clustering distribution (KL divergence / DEC loss).
* **Dimensionality Reduction:** PCA preprocessing of the brain connectivity features (512 components) prior to encoding, with UMAP visualization of learned embeddings.
* **Robust Training Pipeline:** Independent autoencoder pretraining → K-Means cluster initialization → joint training with encoder freezing warmup and early stopping.
* **Statistical Validation:** Permutation-based F-tests with Benjamini-Hochberg FDR correction to assess behavioral enrichment per cluster; comparison against linear CCA baseline.

## 🏗️ Model Architecture

```
X (Brain Connectivity) ──► Encoder_X ──► z_x (20-dim) ──► Decoder_X ──► X̂
                                              │
                                         [Concat] ──► ClusteringLayer (DEC) ──► Q (soft assignments)
                                              │
Y (Behavioral Scores)  ──► Encoder_Y ──► z_y (4-dim)  ──► Decoder_Y ──► Ŷ
```

- **Encoder X:** Linear(512→256) → BN → ReLU → Linear(256→128) → BN → ReLU → Linear(128→20)
- **Encoder Y:** Linear(11→8) → ReLU → Linear(8→4)
- **Clustering Layer:** Student's t-distribution soft assignments over concatenated latent (24-dim)

## 📊 Results (CBCL, k=3)

| k | Silhouette | CCA Corr | Min Cluster Size | Verdict |
|---|-----------|----------|-----------------|---------|
| **3** | **0.2402** | 0.5683 | 124 | **Best overall** |
| 4 | 0.1639 | 0.6464 | 115 | Good alternative |
| 6 | 0.1463 | 0.6628 | 53 | Too fragmented |
| 10 | 0.1316 | 0.6690 | 7 | Too many |

**k=3 cluster profiles:**

| Cluster | n | Profile |
|---------|---|---------|
| 0 | ~700 | Low severity (scores ≈ 50, normative) |
| 1 | ~120 | High severity (scores ≈ 65, clinical range) |
| 2 | ~300 | Low-mild (scores ≈ 52, subclinical) |

All 11 CBCL subscales showed significant between-cluster differences (FDR-corrected permutation tests). Deep CCA-DEC exceeded the linear CCA baseline in total canonical correlation.

## 📁 Repository Structure

```
├── Multimodal-Deep-Canonical-Correlation-Analysis-with-Clustering.ipynb   # Main notebook
└── README.md
```

## ⚙️ Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install torch numpy pandas scikit-learn scipy statsmodels matplotlib umap-learn nilearn
   ```

2. **Data:** Place `.npz` files (`processed_cbcl.npz` or `processed_nihtoolbox.npz`) in your data path and update `npz_file_path` in the notebook. Each file should contain arrays `X` (connectivity features), `Y` (behavioral scores), and `dem` (demographics).

3. **Run:** Execute cells sequentially in the notebook:
   - Imports & configuration (`Config` class)
   - Load and preview data
   - Define model architecture
   - Preprocess data (StandardScaler + PCA)
   - Pretrain autoencoders
   - Initialize clusters (K-Means)
   - Joint training
   - Evaluate and visualize

4. **Key hyperparameters** (in `Config`):
   | Parameter | Default | Description |
   |-----------|---------|-------------|
   | `n_clusters` | 3 | Number of clusters |
   | `pca_components` | 512 | PCA dims for X |
   | `latent_dim_x` | 20 | X encoder output dim |
   | `latent_dim_y` | 4 | Y encoder output dim |
   | `lambda_cca` | 1.0 | Weight for CCA loss |
   | `lambda_dec` | 0.5 | Weight for DEC loss |
   | `lambda_recon` | 0.1 | Weight for reconstruction loss |
   | `joint_epochs` | 2000 | Max joint training epochs |
   | `patience` | 200 | Early stopping patience |

## 👥 Contributors

| **Woohyeon Her** |
| **Michelle Lee** |
