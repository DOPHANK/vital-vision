"""
Cache manager for vision service results.
"""

from typing import Dict, Any, Optional
import hashlib
import json
import os
from datetime import datetime, timedelta

class CacheManager:
    """Manages caching of vision service results."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_key(self, image_path: str, prompt: str) -> str:
        """Generate cache key from image and prompt."""
        content = f"{image_path}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def get_cached_result(self, cache_key: str) -> Optional[str]:
        """Retrieve cached result if available and not expired."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if self._is_cache_valid(cached_data):
                    return cached_data['result']
        return None
        
    def cache_result(self, cache_key: str, result: str) -> None:
        """Cache the result with timestamp."""
        cache_data = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
            
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached result is still valid (less than 24 hours old)."""
        cache_time = datetime.fromisoformat(cached_data['timestamp'])
        return datetime.now() - cache_time < timedelta(hours=24)
