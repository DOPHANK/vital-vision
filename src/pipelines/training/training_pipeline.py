from sklearn.model_selection import train_test_split
import easyocr
from torch.optim import Adam
import torch
import torch.nn as nn
from src.pipelines.training.model_initialization import initialize_model
from src.pipelines.training.evaluation_pipeline import evaluate_model
from src.models.save_model import save_model, load_model

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
