"""
Model architectures for EGEAT experiments.
"""

from .cnn import SimpleCNN, DCGAN_CNN
from .mlp import MLP_Drebin, MLP_MalwareDetection

__all__ = ['SimpleCNN', 'DCGAN_CNN', 'MLP_Drebin', 'MLP_MalwareDetection']

