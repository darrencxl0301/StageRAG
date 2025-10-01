#stagerag/cache.py
from collections import OrderedDict
import hashlib
from typing import Optional, Dict

class LRUCache:
    """Proper LRU Cache implementation using OrderedDict"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_cache_key(self, model_name: str, prompt: str, step_name: str) -> str:
        """Generate cache key"""
        normalized_prompt = prompt.strip().lower()
        content = f"{model_name}:{step_name}:{normalized_prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, model_name: str, prompt: str, step_name: str) -> Optional[str]:
        """Get cached result with LRU update"""
        key = self.get_cache_key(model_name, prompt, step_name)
        
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hit_count += 1
            return value
        
        self.miss_count += 1
        return None
    
    def set(self, model_name: str, prompt: str, step_name: str, result: str):
        """Set cache with proper LRU eviction"""
        key = self.get_cache_key(model_name, prompt, step_name)
        
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Add new entry
        self.cache[key] = result
        
        # Evict least recently used if over capacity
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove least recently used (FIFO order)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_requests": self.hit_count + self.miss_count,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
