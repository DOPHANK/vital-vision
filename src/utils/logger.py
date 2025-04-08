# src/utils/logger.py

import logging
from pathlib import Path

def setup_logger(name: str, log_dir: str = "logs"):
    """Configure logging with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/{name}.log")
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
