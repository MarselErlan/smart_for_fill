"""
Cache Service (Redis) – Handles caching (get/set) for form analysis (and HTML) using Redis.
"""

import json
import redis
from loguru import logger

# (You can load REDIS_URL from an env var, e.g. os.getenv("REDIS_URL", "redis://localhost:6379/0"))
REDIS_URL = "redis://localhost:6379/0"


class CacheService:
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis = redis.from_url(redis_url)

    def get(self, key: str) -> dict:
        """Retrieve (and deserialize) a cached value (or None if not found)."""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Redis get error (key={key}): {e}")
        return None

    def set(self, key: str, value: dict, ttl_seconds: int = 3600) -> bool:
        """Serialize and store a value (with optional TTL)."""
        try:
            self.redis.set(key, json.dumps(value), ex=ttl_seconds)
            return True
        except Exception as e:
            logger.error(f"Redis set error (key={key}): {e}")
            return False 