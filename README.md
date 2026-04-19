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
**Best result:** Tuned Mini-ResNet achieving ~93% test accuracy.

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

#### Model 4 — Tuned Mini-ResNet (Best Model)

A manual grid search over 8 hyperparameter combinations screened for 20 epochs each:

| Hyperparameter | Values searched |
|---|---|
| Learning rate | `1e-2`, `1e-3`, `1e-4` |
| Batch size | `32`, `64`, `128` |
| Dropout rate | `0.3`, `0.4`, `0.5` |

The best configuration is then retrained from scratch for up to 60 epochs with Cosine Decay and EarlyStopping, producing the final model saved to `models/saved/tuned_mini_resnet.h5`.

---

## Results

Results below are representative; exact numbers depend on hardware and TensorFlow version.

| Model | Test Accuracy | Weighted F1 | Parameters | Key Addition |
|---|---|---|---|---|
| Baseline CNN | ~89% | ~0.89 | ~122 K | — |
| Improved CNN | ~91% | ~0.91 | ~710 K | BatchNorm, Dropout, Augmentation |
| Mini-ResNet | ~92% | ~0.92 | ~2.8 M | Residual connections |
| Tuned Mini-ResNet | ~93% | ~0.93 | ~2.8 M | Hyperparameter optimization |

**Consistently hardest classes:** Shirt, T-shirt/top, and Coat — visually similar garments with overlapping textures.

### Key Observations

1. **Regularization reduced overfitting** — the train/validation accuracy gap dropped from ~5–8 pp (Baseline) to ~1–2 pp (Improved CNN and beyond).
2. **Residual connections enabled deeper training** — 13 convolutional layers trained stably without degradation thanks to skip connections.
3. **GlobalAveragePooling is efficient** — replacing a large dense layer with GAP cut parameter count by ~3× in the classifier head while improving generalization.
4. **Adam + Cosine Decay outperformed SGD + fixed LR** — faster convergence and better final minima.

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

Open `notebooks/demo_best_model.ipynb` in Google Colab (GPU runtime recommended). The notebook will:
1. Clone this repository.
2. Download the Fashion MNIST test set from Kaggle.
3. Load the trained model weights.
4. Let you predict any sample — by index or at random.

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

> *This section will be completed by the team.*
