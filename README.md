# EGEAT: Ensemble Gradient-based Ensemble Adversarial Training

A research implementation of **Ensemble Gradient-based Ensemble Adversarial Training (EGEAT)** for improving adversarial robustness through ensemble methods with geometric regularization.

## Overview

EGEAT combines three key techniques:
1. **Adversarial Training**: Training on adversarially perturbed inputs
2. **Geometric Regularization**: Decorrelating gradient subspaces across ensemble members
3. **Ensemble Smoothing**: Parameter soup regularization for ensemble diversity

## Features

- ✅ **Multiple Model Architectures**: SimpleCNN, DCGAN_CNN, MLP variants
- ✅ **Multiple Datasets**: CIFAR-10, MNIST, DREBIN malware detection
- ✅ **Adversarial Attacks**: FGSM, PGD (L∞ and L2), C&W L2
- ✅ **Comprehensive Evaluation**: Accuracy, ECE, gradient similarity, ensemble variance, transferability
- ✅ **GPU Support**: Automatic GPU detection and optimized memory management
- ✅ **Mixed Precision Training**: Optional FP16 training for faster training
- ✅ **Checkpointing**: Resume training from checkpoints
- ✅ **Logging**: Comprehensive logging to files and console
- ✅ **Visualization**: Automatic generation of figures and plots

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU training)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
```bash
cd EGEAT-Experiments
```

2. **Create a virtual environment** (recommended):
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

## Quick Start

### Basic Usage

Run a complete experiment with default settings:

```bash
python run_experiments.py --dataset cifar10 --epochs 10 --ensemble-size 5
```

### Command-Line Options

```bash
python run_experiments.py \
    --experiment-name my_experiment \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 10 \
    --learning-rate 2e-4 \
    --lambda-geom 0.1 \
    --lambda-soup 0.05 \
    --ensemble-size 5 \
    --epsilon 0.031 \
    --device cuda:0 \
    --save-dir results \
    --mixed-precision
```

### Configuration File

Create a JSON config file for more control:

```json
{
  "experiment_name": "egeat_cifar10",
  "seed": 42,
  "device": null,
  "save_dir": "results",
  "training": {
    "batch_size": 128,
    "learning_rate": 2e-4,
    "epochs": 10,
    "lambda_geom": 0.1,
    "lambda_soup": 0.05,
    "epsilon": 0.031,
    "use_mixed_precision": false
  },
  "model": {
    "model_type": "SimpleCNN",
    "ensemble_size": 5
  },
  "data": {
    "dataset": "cifar10",
    "val_split": 0.1
  }
}
```

Then run:
```bash
python run_experiments.py --config config.json
```

## Project Structure

```
EGEAT-Experiments/
├── src/
│   ├── attacks/          # Adversarial attack implementations
│   │   ├── fgsm.py
│   │   ├── pgd.py
│   │   └── cw_l2.py
│   ├── evaluation/       # Evaluation metrics and analysis
│   │   ├── metrics.py
│   │   ├── gradient_similarity.py
│   │   ├── ensemble_variance.py
│   │   ├── transferability.py
│   │   ├── loss_landscape.py
│   │   └── adversarial_images.py
│   ├── models/           # Model architectures
│   │   ├── cnn.py
│   │   └── mlp.py
│   ├── training/         # Training utilities
│   │   └── train_egeat.py
│   ├── utils/            # Utility functions
│   │   ├── data_loader.py
│   │   ├── device_utils.py
│   │   └── viz_utils.py
│   └── config.py         # Configuration management
├── run_experiments.py     # Main experiment runner
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Usage Examples

### Training on CIFAR-10

```bash
python run_experiments.py \
    --dataset cifar10 \
    --model-type SimpleCNN \
    --ensemble-size 5 \
    --epochs 20 \
    --batch-size 128 \
    --learning-rate 2e-4 \
    --lambda-geom 0.1 \
    --lambda-soup 0.05
```

### Training on MNIST

```bash
python run_experiments.py \
    --dataset mnist \
    --model-type SimpleCNN \
    --ensemble-size 3 \
    --epochs 10 \
    --batch-size 256
```

### Using Mixed Precision Training

```bash
python run_experiments.py \
    --dataset cifar10 \
    --epochs 20 \
    --mixed-precision \
    --batch-size 256
```

## Output Structure

After running an experiment, results are saved in the `save_dir` directory:

```
results/
├── config.json              # Experiment configuration
├── results.json             # Evaluation results
├── checkpoints/             # Model checkpoints
│   ├── model_0_final.pt
│   ├── model_1_final.pt
│   └── ...
├── figures/                 # Generated plots
│   ├── grad_similarity.png
│   ├── ensemble_variance.png
│   ├── transferability.png
│   ├── loss_1d.png
│   ├── loss_2d.png
│   └── adv_examples.png
├── adv_images/             # Adversarial example images
│   ├── orig_0.png
│   ├── adv_0.png
│   └── ...
└── logs/                   # Training logs
    └── experiment_timestamp.log
```

## GPU Setup

### Automatic GPU Detection

The code automatically detects and uses available GPUs. To specify a GPU:

```bash
python run_experiments.py --device cuda:0
```

### Multi-GPU Training

For multi-GPU setups, you can run multiple experiments in parallel:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --device cuda:0 --experiment-name exp_gpu0

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python run_experiments.py --device cuda:0 --experiment-name exp_gpu1
```

### Memory Optimization

- Use `--mixed-precision` for reduced memory usage
- Reduce `--batch-size` if running out of memory
- The code automatically clears GPU cache between models

## Key Parameters

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

## Evaluation Metrics

The code automatically computes:

1. **Accuracy**: Standard classification accuracy
2. **ECE (Expected Calibration Error)**: Calibration quality
3. **Gradient Subspace Similarity**: Cosine similarity between model gradients
4. **Ensemble Variance**: Variance of predictions across ensemble
5. **Transferability Matrix**: How adversarial examples transfer between models
6. **Loss Landscapes**: 1D and 2D loss surface visualizations

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size: `--batch-size 64`
- Use mixed precision: `--mixed-precision`
- Reduce ensemble size: `--ensemble-size 3`

### Slow Training

- Enable mixed precision: `--mixed-precision`
- Increase batch size (if memory allows)
- Use fewer workers: Modify `num_workers` in config

### Import Errors

- Ensure you're in the correct directory
- Activate virtual environment
- Install dependencies: `pip install -r requirements.txt`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{egeat2024,
  title={Ensemble Gradient-based Ensemble Adversarial Training},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository or contact the maintainers.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Adversarial robustness research community
