"""
Configuration management for EGEAT experiments.

This module provides a centralized configuration system with support for
loading from files, command-line arguments, and environment variables.
"""

import os
import json
import argparse
from dataclasses import dataclass, asdict, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 128
    learning_rate: float = 2e-4
    epochs: int = 5
    lambda_geom: float = 0.1
    lambda_soup: float = 0.05
    epsilon: float = 8/255
    use_mixed_precision: bool = False
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = 'SimpleCNN'  # 'SimpleCNN', 'DCGAN_CNN', 'MLP_Drebin', 'MLP_MalwareDetection'
    input_channels: int = 3
    num_classes: int = 10
    ensemble_size: int = 5


@dataclass
class AttackConfig:
    """Adversarial attack configuration."""
    epsilon: float = 8/255
    alpha: float = 2/255
    pgd_iters: int = 10
    cw_iters: int = 50
    cw_c: float = 1e-2
    cw_lr: float = 0.01


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    max_batches: int = 10
    n_bins_ece: int = 15
    loss_landscape_grid_n: int = 21
    loss_landscape_radius: float = 1.0


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: str = 'cifar10'  # 'cifar10', 'mnist', 'drebin'
    val_split: float = 0.1
    augment: bool = True
    data_path: str = './data'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    experiment_name: str = 'egeat_experiment'
    seed: int = 42
    device: Optional[str] = None  # None for auto-detect
    save_dir: str = 'results'
    resume: bool = False
    checkpoint_dir: Optional[str] = None
    
    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        """Create config from dictionary."""
        training = TrainingConfig(**d.pop('training', {}))
        model = ModelConfig(**d.pop('model', {}))
        attack = AttackConfig(**d.pop('attack', {}))
        evaluation = EvaluationConfig(**d.pop('evaluation', {}))
        data = DataConfig(**d.pop('data', {}))
        return cls(
            training=training,
            model=model,
            attack=attack,
            evaluation=evaluation,
            data=data,
            **d
        )
    
    def save(self, path: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON file."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='EGEAT Experiment Runner')
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default='egeat_experiment',
                          help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                          help='Device to use (cuda, cpu, or cuda:0, etc.)')
    parser.add_argument('--save-dir', type=str, default='results',
                          help='Directory to save results')
    parser.add_argument('--resume', action='store_true',
                          help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                          help='Directory containing checkpoints')
    parser.add_argument('--config', type=str, default=None,
                          help='Path to JSON config file')
    
    # Data settings
    parser.add_argument('--dataset', type=str, default='cifar10',
                          choices=['cifar10', 'mnist', 'drebin'],
                          help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=128,
                          help='Batch size')
    parser.add_argument('--val-split', type=float, default=0.1,
                          help='Validation split ratio')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=5,
                          help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                          help='Learning rate')
    parser.add_argument('--lambda-geom', type=float, default=0.1,
                          help='Geometric regularization weight')
    parser.add_argument('--lambda-soup', type=float, default=0.05,
                          help='Ensemble smoothing weight')
    parser.add_argument('--epsilon', type=float, default=8/255,
                          help='Adversarial perturbation radius')
    parser.add_argument('--ensemble-size', type=int, default=5,
                          help='Number of models in ensemble')
    parser.add_argument('--mixed-precision', action='store_true',
                          help='Use mixed precision training')
    
    # Model settings
    parser.add_argument('--model-type', type=str, default='SimpleCNN',
                          choices=['SimpleCNN', 'DCGAN_CNN', 'MLP_Drebin', 'MLP_MalwareDetection'],
                          help='Model architecture')
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create config from command-line arguments."""
    config = ExperimentConfig()
    
    # Experiment settings
    config.experiment_name = args.experiment_name
    config.seed = args.seed
    config.device = args.device
    config.save_dir = args.save_dir
    config.resume = args.resume
    config.checkpoint_dir = args.checkpoint_dir
    
    # Data settings
    config.data.dataset = args.dataset
    config.data.val_split = args.val_split
    
    # Training settings
    config.training.batch_size = args.batch_size
    config.training.epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.lambda_geom = args.lambda_geom
    config.training.lambda_soup = args.lambda_soup
    config.training.epsilon = args.epsilon
    config.training.use_mixed_precision = args.mixed_precision
    
    # Model settings
    config.model.model_type = args.model_type
    config.model.ensemble_size = args.ensemble_size
    
    return config

