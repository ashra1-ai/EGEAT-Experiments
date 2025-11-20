"""
Main experiment runner for EGEAT (Ensemble Gradient-based Ensemble Adversarial Training).

This script orchestrates the complete experimental pipeline:
1. Dataset loading
2. Model initialization
3. Ensemble training with EGEAT
4. Evaluation and metric computation
5. Visualization and result saving
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ExperimentConfig, parse_args, create_config_from_args
from src.models import SimpleCNN, DCGAN_CNN, MLP_Drebin, MLP_MalwareDetection
from src.training.train_egeat import train_egeat_epoch
from src.attacks import fgsm_attack, pgd_attack, cw_l2_attack
from src.evaluation.metrics import accuracy, ece
from src.evaluation.gradient_similarity import gradient_subspace_similarity
from src.evaluation.ensemble_variance import ensemble_variance
from src.evaluation.transferability import transferability_matrix, print_transferability_matrix
from src.evaluation.loss_landscape import scan_1d_loss, scan_2d_loss
from src.evaluation.adversarial_images import generate_adv_examples, save_adv_images
from src.utils.data_loader import DataLoaders
from src.utils.device_utils import get_device, clear_gpu_cache, get_gpu_memory_usage
import src.utils.viz_utils as viz


def setup_logging(save_dir: str, experiment_name: str) -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_models(config: ExperimentConfig, device: torch.device, logger: logging.Logger) -> List[nn.Module]:
    """Initialize ensemble models."""
    logger.info(f"Initializing {config.model.ensemble_size} {config.model.model_type} models...")
    
    models = []
    model_class = {
        'SimpleCNN': SimpleCNN,
        'DCGAN_CNN': DCGAN_CNN,
        'MLP_Drebin': MLP_Drebin,
        'MLP_MalwareDetection': MLP_MalwareDetection
    }.get(config.model.model_type, SimpleCNN)
    
    for i in range(config.model.ensemble_size):
        if config.model.model_type in ['MLP_Drebin', 'MLP_MalwareDetection']:
            # For MLP models, we need input_dim - this should be set based on dataset
            model = model_class()
        else:
            model = model_class(
                input_channels=config.model.input_channels,
                num_classes=config.model.num_classes
            )
        model = model.to(device)
        models.append(model)
    
    logger.info(f"Initialized {len(models)} models on {device}")
    return models


def train_ensemble(
    models: List[nn.Module],
    train_loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
    save_dir: str
) -> Dict:
    """Train ensemble of models using EGEAT."""
    logger.info("Starting ensemble training...")
    
    training_history = {
        'models': [],
        'metrics': []
    }
    
    for i, model in enumerate(models):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model {i+1}/{len(models)}")
        logger.info(f"{'='*60}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        
        model_history = {
            'epoch': [],
            'loss': [],
            'adv_loss': [],
            'geom_loss': [],
            'soup_loss': []
        }
        
        # Collect snapshots from previously trained models for ensemble regularization
        ensemble_snapshots = None
        if i > 0 and config.training.lambda_geom > 0:
            ensemble_snapshots = []
            for j in range(i):
                snapshot_path = os.path.join(save_dir, 'checkpoints', f'model_{j}_final.pt')
                if os.path.exists(snapshot_path):
                    snapshot = torch.load(snapshot_path, map_location=device)
                    ensemble_snapshots.append(snapshot)
        
        for epoch in range(config.training.epochs):
            logger.info(f"Epoch {epoch+1}/{config.training.epochs}")
            
            # Train one epoch
            metrics = train_egeat_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                lambda_geom=config.training.lambda_geom,
                lambda_soup=config.training.lambda_soup,
                epsilon=config.training.epsilon,
                ensemble_snapshots=ensemble_snapshots,
                use_mixed_precision=config.training.use_mixed_precision
            )
            
            model_history['epoch'].append(epoch + 1)
            model_history['loss'].append(metrics['loss'])
            model_history['adv_loss'].append(metrics['adv_loss'])
            model_history['geom_loss'].append(metrics['geom_loss'])
            model_history['soup_loss'].append(metrics['soup_loss'])
            
            logger.info(f"  Loss: {metrics['loss']:.4f}, "
                      f"Adv: {metrics['adv_loss']:.4f}, "
                      f"Geom: {metrics['geom_loss']:.4f}, "
                      f"Soup: {metrics['soup_loss']:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(save_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{i}_epoch_{epoch+1}.pt'))
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{i}_final.pt'))
        
        training_history['models'].append(model_history)
        
        # Clear GPU cache
        clear_gpu_cache()
        if device.type == 'cuda':
            allocated, reserved = get_gpu_memory_usage()
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    return training_history


def evaluate_ensemble(
    models: List[nn.Module],
    test_loader: torch.utils.data.DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    logger: logging.Logger,
    save_dir: str
):
    """Evaluate ensemble models and compute metrics."""
    logger.info("\n" + "="*60)
    logger.info("Evaluating ensemble models...")
    logger.info("="*60)
    
    results = {}
    
    # Individual model accuracies
    logger.info("\nComputing model accuracies...")
    accuracies = []
    ece_scores = []
    for i, model in enumerate(models):
        acc = accuracy(model, test_loader, device=device)
        ece_score = ece(model, test_loader, device=device, n_bins=config.evaluation.n_bins_ece)
        accuracies.append(acc)
        ece_scores.append(ece_score)
        logger.info(f"Model {i+1}: Accuracy={acc:.4f}, ECE={ece_score:.4f}")
    
    results['accuracies'] = accuracies
    results['ece_scores'] = ece_scores
    
    # Gradient subspace similarity
    logger.info("\nComputing gradient subspace similarity...")
    sim_matrix = gradient_subspace_similarity(
        models, test_loader, device=device, max_batches=config.evaluation.max_batches
    )
    viz.save_heatmap(
        sim_matrix,
        os.path.join(save_dir, 'figures', 'grad_similarity.png'),
        title="Gradient Subspace Similarity"
    )
    results['grad_similarity'] = sim_matrix.tolist()
    
    # Ensemble variance
    logger.info("\nComputing ensemble variance...")
    Kvals, vars_ = ensemble_variance(
        models, test_loader, device=device, max_batches=config.evaluation.max_batches
    )
    viz.save_line(
        Kvals, vars_,
        os.path.join(save_dir, 'figures', 'ensemble_variance.png'),
        xlabel="K (Number of Models)",
        ylabel="Variance",
        title="Ensemble Variance"
    )
    results['ensemble_variance'] = {'K': Kvals, 'variance': vars_}
    
    # Transferability matrix
    logger.info("\nComputing transferability matrix...")
    P = transferability_matrix(
        models, test_loader, device=device, max_batches=config.evaluation.max_batches,
        epsilon=config.attack.epsilon, alpha=config.attack.alpha, iters=config.attack.pgd_iters
    )
    print_transferability_matrix(P)
    viz.save_heatmap(
        P,
        os.path.join(save_dir, 'figures', 'transferability.png'),
        title="Transferability Matrix"
    )
    results['transferability'] = P.tolist()
    
    # Adversarial examples
    logger.info("\nGenerating adversarial examples...")
    adv_samples = generate_adv_examples(
        models[0], models[1] if len(models) > 1 else models[0],
        test_loader, device=device, n_samples=16
    )
    save_adv_images(adv_samples, os.path.join(save_dir, 'adv_images'))
    viz.save_image_grid(
        [orig for orig, _ in adv_samples[:16]],
        os.path.join(save_dir, 'figures', 'adv_examples_original.png'),
        nrow=4, title="Original Images"
    )
    viz.save_image_grid(
        [adv for _, adv in adv_samples[:16]],
        os.path.join(save_dir, 'figures', 'adv_examples.png'),
        nrow=4, title="Adversarial Examples"
    )
    
    # Loss landscapes
    logger.info("\nComputing loss landscapes...")
    alphas1D, losses1D = scan_1d_loss(
        models[0], test_loader, device=device,
        grid_n=config.evaluation.loss_landscape_grid_n,
        radius=config.evaluation.loss_landscape_radius
    )
    viz.save_line(
        alphas1D, losses1D,
        os.path.join(save_dir, 'figures', 'loss_1d.png'),
        xlabel="Alpha", ylabel="Loss",
        title="1D Loss Landscape"
    )
    
    alphas2D, betas2D, losses2D = scan_2d_loss(
        models[0], test_loader, device=device,
        grid_n=config.evaluation.loss_landscape_grid_n,
        radius=config.evaluation.loss_landscape_radius
    )
    viz.save_contour(
        alphas2D, betas2D, losses2D,
        os.path.join(save_dir, 'figures', 'loss_2d.png'),
        title="2D Loss Landscape"
    )
    
    return results


def main():
    """Main experiment pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config = ExperimentConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = create_config_from_args(args)
        print("Created config from command-line arguments")
    
    # Set up directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, 'adv_images'), exist_ok=True)
    os.makedirs(os.path.join(config.save_dir, 'checkpoints'), exist_ok=True)
    
    # Set up logging
    logger = setup_logging(config.save_dir, config.experiment_name)
    
    # Save config
    config_path = os.path.join(config.save_dir, 'config.json')
    config.save(config_path)
    logger.info(f"Config saved to {config_path}")
    
    try:
        # Set random seed
        set_seed(config.seed)
        logger.info(f"Random seed set to {config.seed}")
        
        # Get device
        device = get_device(config.device, verbose=True)
        logger.info(f"Using device: {device}")
        
        # Load dataset
        logger.info(f"\nLoading {config.data.dataset} dataset...")
        dataloaders = DataLoaders()
        if config.data.dataset.lower() == 'cifar10':
            train_loader, val_loader, test_loader = dataloaders.get_cifar10_loaders(
                batch_size=config.training.batch_size,
                val_split=config.data.val_split,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                augment=config.data.augment
            )
            config.model.input_channels = 3
            config.model.num_classes = 10
        elif config.data.dataset.lower() == 'mnist':
            train_loader, val_loader, test_loader = dataloaders.get_mnist_loaders(
                batch_size=config.training.batch_size,
                val_split=config.data.val_split,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory
            )
            config.model.input_channels = 1
            config.model.num_classes = 10
        elif config.data.dataset.lower() == 'drebin':
            train_loader, val_loader, test_loader = dataloaders.get_drebin_loaders(
                batch_size=config.training.batch_size,
                val_split=config.data.val_split,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                data_path=config.data.data_path
            )
            config.model.num_classes = 2
        else:
            raise ValueError(f"Unsupported dataset: {config.data.dataset}")
        
        logger.info(f"Dataset loaded. Train batches: {len(train_loader)}, "
                   f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        # Initialize models
        models = load_models(config, device, logger)
        
        # Train ensemble
        training_history = train_ensemble(
            models, train_loader, config, device, logger, config.save_dir
        )
        
        # Evaluate ensemble
        results = evaluate_ensemble(
            models, test_loader, config, device, logger, config.save_dir
        )
        
        # Save results
        results_path = os.path.join(config.save_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'config': config.to_dict(),
                'training_history': training_history,
                'evaluation_results': results
            }, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
        
        logger.info("\n" + "="*60)
        logger.info("Experiment completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\nError during experiment: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

