#!/usr/bin/env python3
"""
🧹 Clear Redis Cache Script
Clears all cached form analyses to force fresh analysis with improved prompts
"""

import redis
import sys

def clear_cache():
    """Clear all Redis cache entries"""
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connection
        r.ping()
        print("✅ Connected to Redis")
        
        # Get all keys
        keys = r.keys("*")
        print(f"📊 Found {len(keys)} cached entries")
        
        if keys:
            # Show some example keys
            print("🔍 Sample keys:")
            for key in keys[:5]:
                print(f"   - {key}")
            if len(keys) > 5:
                print(f"   ... and {len(keys) - 5} more")
            
            # Clear all keys
            deleted = r.flushall()
            print(f"🧹 Cleared all cache entries: {deleted}")
        else:
            print("📭 No cache entries found")
            
        print("✅ Cache clearing completed successfully!")
        
    except redis.ConnectionError:
        print("❌ Could not connect to Redis. Make sure Redis is running.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🧹 Redis Cache Cleaner")
    print("=" * 50)
    clear_cache() 