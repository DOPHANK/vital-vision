"""
Evaluation Pipeline Module

This module provides functionality for evaluating OCR models, including
metrics calculation and performance assessment.

Functions:
    evaluate_model: Evaluate model performance on validation data
    calculate_metrics: Calculate various performance metrics
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from logging import logger

def evaluate_model(model: Any, validation_data: List[Tuple[Any, Any]]) -> Dict[str, float]:
    """
    Evaluate model performance on validation data.

    Args:
        model (Any): The OCR model to evaluate
        validation_data (List[Tuple[Any, Any]]): List of (input, target) pairs

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    predictions, labels = [], []
    
    for image, label in validation_data:
        try:
            prediction = model.readtext(image)
            predictions.append(prediction)
            labels.append(label)
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            continue
    
    metrics = calculate_metrics(predictions, labels)
    return metrics

def calculate_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Calculate various performance metrics.

    Args:
        predictions (List[str]): Model predictions
        labels (List[str]): Ground truth labels

    Returns:
        Dict[str, float]: Dictionary containing calculated metrics
    """
    try:
        # Calculate basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # Calculate character error rate (CER)
        cer = calculate_cer(predictions, labels)
        
        # Calculate word error rate (WER)
        wer = calculate_wer(predictions, labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cer': cer,
            'wer': wer
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'cer': 1.0,
            'wer': 1.0
        }

def calculate_cer(predictions: List[str], labels: List[str]) -> float:
    """
    Calculate Character Error Rate (CER).

    Args:
        predictions (List[str]): Model predictions
        labels (List[str]): Ground truth labels

    Returns:
        float: Character Error Rate
    """
    total_chars = sum(len(label) for label in labels)
    total_errors = sum(
        levenshtein_distance(pred, label)
        for pred, label in zip(predictions, labels)
    )
    return total_errors / total_chars if total_chars > 0 else 1.0

def calculate_wer(predictions: List[str], labels: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).

    Args:
        predictions (List[str]): Model predictions
        labels (List[str]): Ground truth labels

    Returns:
        float: Word Error Rate
    """
    total_words = sum(len(label.split()) for label in labels)
    total_errors = sum(
        levenshtein_distance(pred.split(), label.split())
        for pred, label in zip(predictions, labels)
    )
    return total_errors / total_words if total_words > 0 else 1.0

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.

    Args:
        s1 (str): First string
        s2 (str): Second string

    Returns:
        int: Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

class EvaluationPipeline:
    def __init__(self):
        # Add metric tracking
        self.metric_tracker = MetricTracker()
        
        # Add visualization support
        self.visualizer = MetricVisualizer()
        
        # Add A/B testing
        self.ab_tester = ABTester()
        
        # Add performance profiling
        self.profiler = PerformanceProfiler()
