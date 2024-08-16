import torch

def save_model(model, path):
    """
    Saves the trained model's weights to a file.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model.
    path : str
        The file path where the model weights will be saved.

    Returns:
    --------
    None
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Loads the model weights from a file.

    Parameters:
    -----------
    model : torch.nn.Module
        The model architecture to load weights into.
    path : str
        The file path from where the model weights will be loaded.

    Returns:
    --------
    model : torch.nn.Module
        The model with loaded weights.
    """
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model
