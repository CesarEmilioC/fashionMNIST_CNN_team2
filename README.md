# Fashion MNIST Classification — Progressive CNN Development

**Course:** Neural Networks — Final Project  
**Institution:** Instituto Tecnológico y de Estudios Superiores de Monterrey  
**Team 2**

| Member | ID |
|---|---|
| Cesar Castaño | A00830006 |
| Félix Martínez | A01284040 |
| Raúl Valdés Caballero | A01197480 |
| Diego Axel Marquez | A01707739 |

---

## Project Overview

This project develops a classification system for the **Fashion MNIST** dataset using a progressive series of Convolutional Neural Network architectures. Starting from a minimal baseline, each model introduces specific techniques taught in class — regularization, advanced optimization, and residual connections — to demonstrate their concrete impact on generalization performance.

**Dataset:** Fashion MNIST — 70,000 grayscale 28×28 images across 10 clothing categories.  
**Framework:** TensorFlow / Keras.  
**Best result:** Mini-ResNet with Cosine-Decay achieving **94.36 %** test accuracy.

---

## Repository Structure

```
fashionMNIST_CNN_team2/
├── src/
│   ├── data/
│   │   └── loader.py              # Data download, parsing, normalization, splitting
│   ├── models/
│   │   ├── baseline_cnn.py        # Model 1: bare-bones CNN
│   │   ├── improved_cnn.py        # Model 2: regularized CNN
│   │   └── mini_resnet.py         # Model 3: residual network
│   ├── training/
│   │   └── trainer.py             # Callbacks, LR schedules, history I/O
│   └── utils/
│       ├── visualization.py       # All plotting functions
│       └── evaluation.py          # Metrics, classification report, comparison table
├── scripts/
│   ├── train_all.py               # Full training pipeline (runs all models)
│   └── hyperparameter_search.py   # Manual grid search on Mini-ResNet
├── notebooks/
│   └── demo_best_model.ipynb      # Colab demo — predict any Fashion MNIST sample
├── results/
│   ├── figures/                   # Saved training curves, confusion matrices, etc.
│   └── metrics/                   # JSON histories, CSV summary, HP search results
├── models/
│   └── saved/                     # Trained model weights (.h5)
├── fashion_mnist_cnn.ipynb        # Original exploratory notebook
├── requirements.txt
└── README.md
```

---

## Approach

### Data Pipeline

The raw Fashion MNIST data is downloaded from Kaggle as two CSV files (60 000 training + 10 000 test rows), each row containing a label and 784 pixel values. The preprocessing pipeline:

1. **Reshape** — pixel vectors → 28×28×1 image tensors.
2. **Normalize** — pixel values scaled from `[0, 255]` to `[0.0, 1.0]`.
3. **Stratified split** — training set further divided 80/20 into train and validation, preserving class balance.

The final split is **48 000 train / 12 000 validation / 10 000 test**.

### Model Progression

We built four models in sequence, each adding complexity in a principled way:

#### Model 1 — Baseline CNN

A minimal two-block CNN with no regularization, trained with SGD at a fixed learning rate. This establishes a floor for comparison and surfaces the overfitting problem clearly.

| Layer | Detail |
|---|---|
| Conv2D × 2 | 32 → 64 filters, 3×3, ReLU |
| MaxPool × 2 | 2×2 |
| Dense | 128 → 10 (softmax) |
| Optimizer | SGD (lr=0.01) |

**Limitation:** Training and validation accuracy diverge significantly after a few epochs, indicating overfitting.

---

#### Model 2 — Improved CNN

Addresses Model 1's weaknesses by adding four complementary regularization techniques:

| Technique | Purpose |
|---|---|
| **Batch Normalization** | Stabilizes and accelerates training by normalizing intermediate activations |
| **Dropout** (0.25 conv, 0.5 dense) | Randomly deactivates neurons to reduce co-adaptation |
| **Data Augmentation** | Random flips, rotations, zooms, and translations increase effective dataset size |
| **L2 Weight Decay** | Penalizes large weights, encouraging simpler decision boundaries |
| **Adam Optimizer** | Adaptive per-parameter learning rates for faster convergence |
| **ReduceLROnPlateau** | Halves the LR when validation loss plateaus (patience=3) |
| **EarlyStopping** | Halts training and restores best weights when val loss stagnates (patience=7) |

Three convolutional blocks (32 → 64 → 128 filters) with double-conv per block, followed by a dense head with Dropout.

---

#### Model 3 — Mini-ResNet

Introduces **residual (skip) connections**, the central innovation of He et al. (2016). Each residual block computes:

```
output = F(x) + x
```

where `F(x)` is a two-layer convolution path and `x` is a shortcut that bypasses it. If the shortcut and main path have different dimensions (due to strided convolution), a 1×1 convolution is applied to the shortcut to match dimensions.

**Why this matters:**
- Gradients can flow directly through the skip path, mitigating the **vanishing gradient problem** that limits deep plain networks.
- Residual learning is inherently easier: layers only need to learn a *residual* correction rather than the full mapping.
- Empirically enables training of significantly deeper networks without degradation.

**Architecture:**
- Initial 64-filter conv layer
- Stage 1: 2 residual blocks @ 64 filters, 28×28 spatial resolution
- Stage 2: 2 residual blocks @ 128 filters, 14×14 (stride-2 downsampling)
- Stage 3: 2 residual blocks @ 256 filters, 7×7 (stride-2 downsampling)
- **GlobalAveragePooling2D** — averages each feature map to a single value, replacing the large Flatten+Dense combination. This drastically reduces parameters and overfitting risk.
- Dense(10, softmax) output

**Learning rate schedule:** Cosine Decay — smoothly anneals the LR from the initial value to near-zero over training, helping the optimizer settle into a better minimum than step-wise decay.

---

#### Model 4 — Tuned Mini-ResNet

A manual grid search over 8 hyperparameter combinations screened for 20 epochs each:

| Hyperparameter | Values searched |
|---|---|
| Learning rate | `1e-2`, `1e-3`, `1e-4` |
| Batch size | `32`, `64`, `128` |
| Dropout rate | `0.3`, `0.4`, `0.5` |

The best configuration by 20-epoch validation accuracy (`lr=1e-4, bs=64, dropout=0.4`) was then retrained from scratch for up to 60 epochs with Cosine Decay and EarlyStopping, producing `models/saved/tuned_mini_resnet.h5`.

See the full search results in [`results/metrics/hyperparameter_search.csv`](results/metrics/hyperparameter_search.csv).

---

## Results

Final metrics on the held-out test set (10 000 samples), trained on Google Colab with a T4 GPU.

| Model | Test Accuracy | Weighted F1 | Parameters | Train Time | Key Addition |
|---|---|---|---|---|---|
| Baseline CNN | 89.91 % | 0.900 | 421 K | 1.8 min | — |
| Improved CNN | 92.40 % | 0.925 | 439 K | 11.8 min | BatchNorm, Dropout, Augmentation, L2, Adam |
| **Mini-ResNet** | **94.36 %** | **0.944** | **2.78 M** | **33.1 min** | **Residual connections + Cosine Decay** |
| Tuned Mini-ResNet | 93.13 % | 0.932 | 2.78 M | 47.2 min | Hyperparameter search (60 epochs) |

The full numeric table is stored at [`results/metrics/results_summary.csv`](results/metrics/results_summary.csv) and visualized in [`results/figures/results_bar_comparison.png`](results/figures/results_bar_comparison.png).

**Consistently hardest classes:** Shirt, T-shirt/top, Pullover and Coat — visually similar upper-body garments whose silhouettes overlap in 28×28 grayscale.

### Key Observations

1. **Each architectural change paid off.** Test accuracy climbed monotonically from Baseline (89.91 %) → Improved (92.40 %) → Mini-ResNet (94.36 %). The biggest single jump (+1.96 pp) came from adding residual connections and moving to Global Average Pooling.
2. **Regularization closed the generalization gap.** The Baseline's train/val curves diverge around epoch 5–8; the Improved CNN and Mini-ResNet track each other within ~1–2 pp for the entire run.
3. **Residual connections enabled deeper training without degradation.** 13 convolutional layers converged stably and without the vanishing-gradient symptoms a plain deep CNN would show at the same depth.
4. **The hyperparameter search did not beat the default Mini-ResNet.** Screening over 20 epochs selected `lr=1e-4` because it produced the smoothest short-horizon validation curve, but at 60 epochs the original `lr=1e-3 + CosineDecay` found a better minimum (+1.23 pp test accuracy). This is a useful lesson on the limits of short-horizon screening — the LR ranking at 20 epochs is not the LR ranking at 60.
5. **Compute scales faster than accuracy.** Going from Improved CNN to Mini-ResNet multiplied both parameter count (~6×) and training time (~3×) for a +1.96 pp gain — a diminishing-returns pattern typical of this dataset at this depth.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

A **Kaggle account** is required to download the dataset on first run (uses `kagglehub`). Alternatively, point the scripts at a local directory containing the CSVs.

### Train all models

```bash
# Download data from Kaggle automatically:
python scripts/train_all.py

# Use local CSV files:
python scripts/train_all.py --data-dir /path/to/csv/dir

# On CPU (suppress GPU warning):
python scripts/train_all.py --no-gpu-check
```

Outputs are written to `results/figures/`, `results/metrics/`, and `models/saved/`.

### Hyperparameter search only

```bash
python scripts/hyperparameter_search.py --epochs 20
```

### Colab Demo

Open [`notebooks/demo_best_model.ipynb`](notebooks/demo_best_model.ipynb) in Google Colab (GPU runtime recommended). The notebook will:
1. Clone this repository.
2. Download the Fashion MNIST test set from Kaggle.
3. Load the trained Mini-ResNet weights from [`models/saved/mini_resnet.h5`](models/saved/mini_resnet.h5).
4. Let you predict any sample — by index or at random — and visualize the full probability distribution across the 10 classes.

A second notebook, [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), reproduces the full training pipeline end-to-end in Colab.

---

## References

1. Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. arXiv:1708.07747.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR, pp. 770–778.
3. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML, pp. 448–456.
4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15(1), pp. 1929–1958.
5. Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
6. Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR.
7. Lin, M., Chen, Q., & Yan, S. (2014). *Network In Network*. ICLR.
8. Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(1).

---

## Conclusions

The progressive development pipeline produced a clear, measurable payoff for every design decision we borrowed from the course material. Starting from a bare-bones CNN at 89.91 % test accuracy, each layer of complexity — regularization, then residual connections — lifted performance by roughly **2 pp** until we converged on **94.36 %** with the Mini-ResNet. The hardest remaining errors concentrate on visually ambiguous upper-body garments (Shirt / T-shirt / Pullover / Coat), which is consistent with the published Fashion-MNIST error distribution and suggests that raw 28×28 grayscale pixels are near the ceiling of what a general-purpose CNN of this size can extract.

The most instructive finding was negative: our **hyperparameter search did not produce the best final model**. Screening 8 configurations for 20 epochs picked a conservative learning rate (`1e-4`) that minimized validation loss over a short horizon, but when we retrained it for 60 epochs the optimizer never reached the minimum that the default `lr=1e-3 + CosineDecay` found. The takeaway is pragmatic — short-horizon HP screening is a useful filter but can systematically undervalue schedules that need longer to pay off, and validating the "winner" against the un-tuned baseline at full training length is worth the extra compute.

On the architectural side, two ideas from the course turned out to be doing most of the work: **residual connections**, which were responsible for the single largest accuracy jump (+1.96 pp) and made the deeper network trainable in the first place; and **Global Average Pooling**, which let us grow the convolutional stack to 2.78 M parameters without ballooning the classifier head or overfitting. Data augmentation, Batch Normalization, Dropout and L2 also each contributed measurably to closing the generalization gap in Model 2.

If we had more compute to invest, the obvious next steps would be: a longer-horizon HP search (40–60 epochs per config), an ensemble of independently seeded Mini-ResNets, and targeted augmentation aimed at the Shirt/T-shirt confusion pair. None of these are conceptually new, but on Fashion-MNIST they are the known routes past the ~95 % threshold.
