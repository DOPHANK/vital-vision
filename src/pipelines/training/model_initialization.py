
import torch
import torch.nn as nn
import torchvision.models as models

def initialize_model(num_classes=10, use_pretrained=True):
    '''
    Initializes the OCR model, possibly using a pre-trainaed model as a starting point.

    Parameters:
    -----------
    num_classes : int
        The number of output classes for the model.
    use_pretrained : bool
        If True, uses a model pre-trained on a large dataset (e.g., ImageNet).

    Returns:
    --------
    model : torch.nn.Module
        The initialized model ready for training.
    '''
    # Example: Using ResNet as the backbone for OCR
    model = models.resnet50(pretrained=use_pretrained)

    # Modify the final fully connected layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # If we are doing fine-tuning, we freeze all layers except the final one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
    