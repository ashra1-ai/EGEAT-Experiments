"""
Evaluation metrics and analysis tools.
"""

from .metrics import accuracy, ece, nll_loss, gradient_cosine_similarity
from .gradient_similarity import gradient_subspace_similarity, gradient_subspace_distance
from .ensemble_variance import ensemble_variance
from .transferability import transferability_matrix
from .loss_landscape import scan_1d_loss, scan_2d_loss
from .adversarial_images import generate_adv_examples, save_adv_images

__all__ = [
    'accuracy', 'ece', 'nll_loss', 'gradient_cosine_similarity',
    'gradient_subspace_similarity', 'gradient_subspace_distance',
    'ensemble_variance', 'transferability_matrix',
    'scan_1d_loss', 'scan_2d_loss',
    'generate_adv_examples', 'save_adv_images'
]

