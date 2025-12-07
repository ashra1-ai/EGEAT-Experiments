# 🧠 Exact Geometric Ensemble Adversarial Training (EGEAT)

> **A Unified Framework for Exact Optimization, Gradient-Space Geometry, and Ensemble Robustness**

This repository contains the complete implementation of **EGEAT — Exact Geometric Ensemble Adversarial Training**, a unified adversarial learning framework that combines **exact optimization**, **gradient-space regularization**, and **ensemble smoothing** into one theoretically grounded formulation.  

📄 *Based on the research paper:*  

**"Exact Geometric Ensemble Adversarial Training (EGEAT): A Unified Framework for Robust Optimization and Gradient-Space Regularization" (2024)**

---

## 📘 Abstract

EGEAT unifies three complementary components of adversarial robustness — **analytic perturbations**, **gradient-space geometry**, and **ensemble variance smoothing**.  

By deriving **closed-form adversarial perturbations** from convex duality, **decorrelating gradient subspaces** across models, and introducing **parameter-space smoothing**, EGEAT achieves robustness and stability without relying on inefficient iterative PGD updates.

---

## 🧩 Method Overview

EGEAT integrates three mechanisms that reinforce each other:

### 1. **Exact Adversarial Optimization**
   - Uses convex duality to derive analytic perturbations
   - Removes the need for iterative inner maximization (PGD)
   - Guarantees first-order optimality for $\delta^*$

### 2. **Geometric Regularization**
   - Penalizes gradient alignment among ensemble members
   - Promotes diversity through subspace orthogonalization
   - Reduces adversarial transfer between models

### 3. **Ensemble Smoothing**
   - "Parameter soup" regularization to stabilize optimization
   - Reduces curvature of the loss landscape
   - Improves calibration and ensemble consistency

---

## 🧮 Theoretical Formulation

Let $\{f_{\theta_k}\}_{k=1}^K$ be an ensemble of models.

\[
\mathcal{L}_{\text{EGEAT}} =
\mathbb{E}_{(x,y)} \Big[
  \frac{1}{K} \sum_{k=1}^K
  \mathcal{L}_{\text{CE}}(f_{\theta_k}(x + \delta_k^\star), y)
  + \lambda_1 \mathcal{L}_{\text{geom}}
  + \lambda_2 \mathcal{L}_{\text{soup}}
\Big]
\]

where:

- $\delta_k^\star$: analytic adversarial perturbation (closed-form via convex duality)  
- $\mathcal{L}_{\text{geom}} = \sum_{i<j} \cos^2(\nabla_x f_{\theta_i}, \nabla_x f_{\theta_j})$: geometric decorrelation  
- $\mathcal{L}_{\text{soup}} = \| \theta_k - \bar{\theta} \|^2$: ensemble variance penalty  

---

## 📊 Visual Results

### 🧮 Variance vs Ensemble Size — *Parameter Soup Stability*

EGEAT exhibits reduced variance with increasing ensemble size, confirming stability through parameter-space smoothing. The 3D surface plot demonstrates how variance decays logarithmically as both ensemble size and regularization strength increase.

![Variance vs Ensemble Size](figures/fig5_variance_surface.png)

*Figure 1: 3D surface plot showing variance reduction with ensemble size K and regularization parameter. Lower variance (dark blue) indicates greater ensemble stability.*

---

### 🌌 3D Gradient Constellation (PCA) — *Decorrelated Subspaces*

Principal component analysis of input gradients across models shows decorrelated subspaces for EGEAT compared to PGD and Soup baselines. The visualization reveals how geometric regularization promotes gradient diversity in the principal component space.

![3D Gradient Constellation](figures/gradient_constellation_3d.png)

*Figure 2: PCA projection of gradient vectors onto three principal components. EGEAT (light blue), Soup (orange), and PGD (teal) show distinct clustering patterns, with EGEAT exhibiting greater subspace decorrelation.*

---

### 🏔️ Loss Landscape Around EGEAT Model — *Flatter Geometry*

2D loss surface visualization showing EGEAT's flatter, more isotropic curvature, indicating improved geometric stability. The contour plot reveals a smoother optimization basin with reduced sharp minima.

![Loss Landscape](figures/loss_landscape_showcase.png)

*Figure 3: Cross-entropy loss landscape in parameter space (α, β directions). The dark blue region indicates the minimum loss basin, demonstrating EGEAT's smoother optimization landscape compared to standard adversarial training.*

---

### 🔁 Adversarial Transfer Graph — *Reduced Cross-Model Transferability*

Visualization of inter-model adversarial transfer rates. Lower edge weights indicate stronger geometric independence and reduced transferability between ensemble members.

![Adversarial Transfer Graph](figures/transfer_graph.png)

*Figure 4: Transfer graph showing adversarial example transfer rates between Soup, PGD, and EGEAT models. Lower values (closer to 0) indicate better robustness and reduced cross-model attack transferability.*

---

### 📈 Training Progress — *Convergence Dynamics*

Training dynamics showing loss reduction and accuracy improvement over epochs, demonstrating stable convergence of the EGEAT framework.

![PGD Training Progress](figures/pgd_training_progress.png)

*Figure 5: Training loss (left) and validation accuracy (right) over 20 epochs. EGEAT demonstrates consistent convergence with decreasing loss and improving robustness.*

---

## 🚀 Features

- ✅ **Closed-form exact adversarial perturbations** — No iterative PGD needed
- ✅ **Gradient-space orthogonalization regularizer** — Promotes ensemble diversity
- ✅ **Parameter-soup smoothing** — Stabilizes ensemble optimization
- ✅ **Multi-dataset support** — CIFAR-10, MNIST, and DREBIN benchmarks
- ✅ **Comprehensive attack suite** — PGD, FGSM, and C&W attacks
- ✅ **Rich visualization suite** — Gradient, loss, and transfer metrics
- ✅ **Production-ready** — Mixed-precision and GPU-optimized training

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU training)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd EGEAT-Experiments
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🚀 Training Command

### Basic Usage

```bash
python run_experiments.py \
  --dataset cifar10 \
  --model-type SimpleCNN \
  --ensemble-size 5 \
  --lambda-geom 0.1 \
  --lambda-soup 0.05 \
  --epochs 20 \
  --epsilon 0.031 \
  --mixed-precision
```

### Advanced Configuration

```bash
python run_experiments.py \
  --experiment-name egeat_cifar10 \
  --dataset cifar10 \
  --model-type SimpleCNN \
  --ensemble-size 5 \
  --batch-size 128 \
  --epochs 20 \
  --learning-rate 2e-4 \
  --lambda-geom 0.1 \
  --lambda-soup 0.05 \
  --epsilon 0.031 \
  --device cuda:0 \
  --save-dir results \
  --mixed-precision
```

---

## 📁 Project Structure

```
EGEAT-Experiments/
├── src/
│   ├── attacks/              # FGSM, PGD, CW
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   └── cw_l2.py
│   ├── evaluation/           # Metrics, loss, similarity, transfer
│   │   ├── metrics.py
│   │   ├── gradient_similarity.py
│   │   ├── ensemble_variance.py
│   │   ├── transferability.py
│   │   ├── loss_landscape.py
│   │   └── adversarial_images.py
│   ├── models/               # CNN, MLP
│   │   ├── cnn.py
│   │   └── mlp.py
│   ├── training/             # Core EGEAT training loop
│   │   └── train_egeat.py
│   ├── utils/                # Visualization and helpers
│   │   ├── data_loader.py
│   │   ├── device_utils.py
│   │   └── viz_utils.py
│   └── config.py
├── results/
│   ├── figures/              # Auto-generated figures
│   ├── logs/                 # Training logs
│   └── checkpoints/          # Model weights
├── run_experiments.py        # Main experiment runner
├── requirements.txt
└── README.md
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Gradient Subspace Similarity** | Measures cosine overlap between model gradients |
| **Ensemble Variance** | Quantifies ensemble dispersion (stability) |
| **Transferability** | Fraction of attacks transferring between models |
| **Loss Landscape Curvature** | Visual curvature of optimization basin |
| **ECE (Calibration)** | Expected calibration error under perturbations |
| **Robust Accuracy** | Classification accuracy under adversarial attacks |

---

## 🧩 Key Findings

- **Geometric regularization** reduces adversarial transferability by **6–8%**
- **Ensemble variance** decays logarithmically with ensemble size
- **Loss basins** under EGEAT are smoother and more isotropic
- **Gradient PCA** confirms subspace decorrelation among ensemble members
- **Closed-form perturbations** achieve comparable robustness to iterative PGD with **3–5× faster training**

---

## 🔬 Experimental Results (Summary)

| Dataset | Model | Attack | Robust Accuracy (PGD-20) | Δ vs PGD |
|---------|-------|--------|--------------------------|----------|
| CIFAR-10 | SimpleCNN | PGD | **0.31** | **+4.5%** |
| MNIST | MLP | FGSM | **0.94** | **+2.3%** |
| DREBIN | MLP | L∞ | **0.88** | **+3.7%** |

---

## 🎯 Key Parameters

### Training Parameters

- `--lambda-geom`: Weight for geometric regularization (default: 0.1)
  - Higher values encourage more diverse gradient subspaces
- `--lambda-soup`: Weight for ensemble smoothing (default: 0.05)
  - Higher values encourage parameters closer to ensemble mean
- `--epsilon`: Adversarial perturbation radius (default: 8/255 ≈ 0.031)
  - Standard value for CIFAR-10/MNIST

### Model Parameters

- `--ensemble-size`: Number of models in ensemble (default: 5)
  - More models = better robustness but longer training time
- `--model-type`: Architecture choice
  - `SimpleCNN`: Lightweight CNN for CIFAR-10/MNIST
  - `DCGAN_CNN`: Deeper CNN variant
  - `MLP_Drebin`: MLP for DREBIN malware detection

---

## 🧭 Future Directions

- **Higher-order curvature modeling**: Incorporate Hessian-aware terms
- **Adaptive ensembles**: Bayesian averaging or diffusion-based sampling
- **Cross-domain transfer**: Extend to multimodal or sequential RL tasks
- **Theoretical guarantees**: Formal convergence and robustness certificates
- **Scalability**: Efficient training for large-scale vision transformers

---
