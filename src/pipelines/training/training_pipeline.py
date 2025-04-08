"""
Training Pipeline Module

This module provides a comprehensive training pipeline for OCR models, including
model initialization, training loops, validation, and early stopping mechanisms.

Classes:
    TrainingPipeline: Main training pipeline class
    EarlyStopping: Early stopping mechanism for training
    ConfigValidationError: Custom exception for configuration validation failures

Functions:
    compute_loss: Compute loss between model output and ground truth
    adjust_learning_rate: Adjust learning rate based on epoch
    train_model: Legacy training function (deprecated)
"""

from sklearn.model_selection import train_test_split
import easyocr
from torch.optim import Adam
import torch
import torch.nn as nn
from src.pipelines.training.model_initialization import initialize_model
from src.pipelines.training.evaluation_pipeline import evaluate_model
from src.models.save_model import save_model, load_model
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from loguru import logger

def compute_loss(output, label):
    """
    Computes the loss between the model's output and the ground truth labels.

    Parameters:
    -----------
    output : torch.Tensor
        The output from the model (predictions).
    label : torch.Tensor
        The ground truth labels.

    Returns:
    --------
    torch.Tensor
        The computed loss.
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, label)
    return loss

def load_data(directory):
    # Load and preprocess data from the directory
    pass


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_factor=0.1, decay_epoch=10):
    """
    Adjusts the learning rate based on the epoch number.
    
    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer being used in training.
    epoch : int
        The current epoch number.
    initial_lr : float
        The initial learning rate.
    decay_factor : float
        Factor by which the learning rate will be reduced.
    decay_epoch : int
        The number of epochs after which the learning rate will be reduced.
        
    Returns:
    --------
    None
    """
    lr = initial_lr * (decay_factor ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Learning rate adjusted to {lr}")

# src/ocr/training_pipeline.py

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Initializes the early stopping mechanism.
        
        Parameters:
        -----------
        patience : int
            Number of epochs to wait for an improvement before stopping.
        min_delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Checks whether training should be stopped early.
        
        Parameters:
        -----------
        val_loss : float
            The current validation loss.
        
        Returns:
        --------
        bool
            Whether to stop training early.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_model(train_data, val_data, epochs=50, initial_lr=0.001, patience=5):
    """
    Trains the OCR model and integrates learning rate adjustment and early stopping.
    
    Parameters:
    -----------
    train_data : Dataset
        The training dataset.
    val_data : Dataset
        The validation dataset.
    epochs : int
        Number of epochs to train the model.
    initial_lr : float
        The initial learning rate.
    patience : int
        Patience for early stopping.
    
    Returns:
    --------
    None
    """
    model = initialize_model()  # Hypothetical function to initialize your OCR model
    optimizer = Adam(model.parameters(), lr=initial_lr)
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        for image, label in train_data:
            optimizer.zero_grad()
            output = model(image)
            loss = compute_loss(output, label)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, label in val_data:
                output = model(image)
                val_loss += compute_loss(output, label).item()
        val_loss /= len(val_data)
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")
        
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, initial_lr)

        # Check early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break

    # Save the final model
    save_model(model, '/models/ocr_model.pth')
    

# Example usage    
# train_model(train_data, val_data, epochs=50, initial_lr=0.001, patience=5)

class ConfigValidationError(Exception):
    """Custom exception for training configuration validation failures."""
    pass

class TrainingPipeline:
    """
    Comprehensive training pipeline for OCR models.

    This class provides a complete training pipeline with features like:
    - Configuration validation
    - Model initialization
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint saving
    - Metrics tracking

    Attributes:
        config (Dict[str, Any]): Training configuration
        model (nn.Module): The OCR model
        optimizer (torch.optim.Optimizer): Model optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        metrics (Dict[str, List[float]]): Training metrics history

    Methods:
        train: Main training loop
        _train_epoch: Single epoch training
        _validate: Model validation
        _save_checkpoint: Save model checkpoint
        _log_metrics: Log training metrics
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the training pipeline.

        Args:
            config (Dict[str, Any]): Training configuration dictionary

        Raises:
            ConfigValidationError: If required configuration fields are missing
        """
        self.config = config
        self._validate_config()
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        self.distributed_trainer = DistributedTrainer()
        self.experiment_tracker = ExperimentTracker()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.model_pruner = ModelPruner()

    def _validate_config(self) -> None:
        """
        Validate training configuration.

        Raises:
            ConfigValidationError: If required fields are missing
        """
        required_fields = ['learning_rate', 'epochs', 'batch_size']
        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            raise ConfigValidationError(f"Missing required fields: {missing_fields}")

    def _initialize_model(self) -> nn.Module:
        """
        Initialize the OCR model.

        Returns:
            nn.Module: Initialized model
        """
        return initialize_model(
            num_classes=self.config.get('num_classes', 10),
            use_pretrained=self.config.get('use_pretrained', True)
        )

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """
        Initialize the model optimizer.

        Returns:
            torch.optim.Optimizer: Initialized optimizer
        """
        return Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

    def _initialize_scheduler(self) -> torch.optim.lr_scheduler:
        """
        Initialize the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler: Initialized scheduler
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

    def train(self, train_data: DataLoader, val_data: DataLoader) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_data (DataLoader): Training data loader
            val_data (DataLoader): Validation data loader

        Returns:
            Dict[str, Any]: Training metrics history
        """
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 5)
        )

        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss = self._train_epoch(train_data)
            
            # Validation phase
            val_loss = self._validate(val_data)
            
            # Update metrics
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log metrics
            self._log_metrics(epoch)
            
            # Save checkpoint
            self._save_checkpoint(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Check early stopping
            if early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break

        return self.metrics

    def _train_epoch(self, train_data: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_data (DataLoader): Training data loader

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_data):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = compute_loss(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % self.config.get('log_interval', 10) == 0:
                logger.info(f"Training batch {batch_idx}: loss = {loss.item():.4f}")
        
        return total_loss / len(train_data)

    def _validate(self, val_data: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_data (DataLoader): Validation data loader

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_data:
                output = self.model(data)
                loss = compute_loss(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_data)

    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics
        }
        
        path = f"{self.config.get('checkpoint_dir', 'checkpoints')}/model_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def _log_metrics(self, epoch: int) -> None:
        """
        Log training metrics.

        Args:
            epoch (int): Current epoch number
        """
        logger.info(
            f"Epoch {epoch + 1}/{self.config['epochs']}: "
            f"train_loss = {self.metrics['train_loss'][-1]:.4f}, "
            f"val_loss = {self.metrics['val_loss'][-1]:.4f}, "
            f"lr = {self.metrics['learning_rates'][-1]:.6f}"
        )
