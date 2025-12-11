"""
Text Summarization Project - Iterative Refinement System
"""

__version__ = "0.1.0"

# Import modules for easy access
from .preprocessor import VietnamesePreprocessor
from .data_loader import VietNewsDataset
from .extractive import ExtractiveModel
from .abstractive import AbstractiveModel

__all__ = [
    'VietnamesePreprocessor',
    'VietNewsDataset',
    'ExtractiveModel',
    'AbstractiveModel',
]









