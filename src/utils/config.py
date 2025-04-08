from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml

@dataclass
class OCRConfig:
    model: str
    languages: List[str]
    max_retries: int
    cache_enabled: bool
    
@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int
    batch_size: int
    early_stopping_patience: int
    
class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load and validate configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return self._validate_config(config)

class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self):
        self.config_validator = ConfigValidator()
        self.config_loader = ConfigLoader()
        self.config_updater = ConfigUpdater()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration."""
        config = self.config_loader.load(config_path)
        self.config_validator.validate(config)
        return config
