# src/ocr/tune_hyperparameters.py

from src.ocr.ocr_service import OCRService
from sklearn.model_selection import ParameterGrid

# Define parameter grid for tuning
param_grid = {
    'language': [['en'], ['vi'], ['th']],
    'gpu': [True, False],
    'detector': [True, False],
    'recognizer': [True, False],
}

def run_tuning():
    for params in ParameterGrid(param_grid):
        print(f"Testing with params: {params}")
        ocr_service = OCRService(model='easyocr', **params)
        # Run tests on validation dataset
        # Evaluate performance and log results
        # ...
        print("Finished testing with params:", params)

if __name__ == "__main__":
    run_tuning()
