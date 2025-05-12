"""
Configuration for vision service.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_id: str = "microsoft/Phi-3-vision-128k-instruct"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    use_gpu: bool = True
    trust_remote_code: bool = True

@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    batch_size: int = 1
    timeout: int = 30
    retry_attempts: int = 3
    cache_results: bool = True
