import numpy as np
from typing import List, Set, Dict
from collections import deque

class CacheManager:
    """Manage service caching with various replacement policies"""
    
    def __init__(self, capacity: int, policy: str = 'LRU'):
        self.capacity = capacity
        self.policy = policy
        self.cache: Set[int] = set()
        self.access_history = deque(maxlen=capacity)
        self.popularity_scores: Dict[int, float] = {}
        
    def can_cache(self, service_id: int, service_size: float,
                  available_space: float) -> bool:
        """Check if service can be cached"""
        return service_size <= available_space
    
    def add_service(self, service_id: int, popularity: float):
        """Add service to cache"""
        if len(self.cache) >= self.capacity:
            # Apply replacement policy
            if self.policy == 'LRU':
                self._evict_lru()
            elif self.policy == 'LFU':
                self._evict_lfu()
            elif self.policy == 'popularity':
                self._evict_least_popular()
                
        self.cache.add(service_id)
        self.popularity_scores[service_id] = popularity
        self.access_history.append(service_id)
        
    def _evict_lru(self):
        """Evict least recently used service"""
        if self.access_history:
            # Find least recently used that's in cache
            for service in self.access_history:
                if service in self.cache:
                    self.cache.remove(service)
                    break
                    
    def _evict_lfu(self):
        """Evict least frequently used service"""
        frequency = {}
        for service in self.access_history:
            if service in self.cache:
                frequency[service] = frequency.get(service, 0) + 1
                
        if frequency:
            least_frequent = min(frequency, key=frequency.get)
            self.cache.remove(least_frequent)
            
    def _evict_least_popular(self):
        """Evict least popular service"""
        if self.cache and self.popularity_scores:
            cache_popularity = {
                sid: self.popularity_scores.get(sid, 0) 
                for sid in self.cache
            }
            least_popular = min(cache_popularity, key=cache_popularity.get)
            self.cache.remove(least_popular)
            
    def has_service(self, service_id: int) -> bool:
        """Check if service is cached"""
        return service_id in self.cache
    
    def get_cached_services(self) -> List[int]:
        """Get list of cached services"""
        return list(self.cache)