#!/usr/bin/env python3
"""
Smart Form Fill API - Main Entry Point
Initializes the FastAPI application, includes modular routers, and handles startup tasks.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Import modular routers
from app.endpoints import meta, vector, form

# Load environment variables
load_dotenv()

# Environment variable validation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_form_filler")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

def clear_redis_cache_on_startup():
    """Clear Redis cache on server startup to ensure fresh form analysis."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        
        keys = r.keys("*")
        if keys:
            r.flushall()
            logger.info(f"🧹 Cleared {len(keys)} cached entries on startup")
        else:
            logger.info("📭 No cache entries found on startup")
            
    except redis.ConnectionError:
        logger.warning("⚠️  Redis not available - cache clearing skipped")
    except Exception as e:
        logger.warning(f"⚠️  Cache clearing failed: {e}")

# Clear cache on startup
clear_redis_cache_on_startup()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Form Fill API",
    description="Vector Database Management + Form Auto-Fill Pipeline",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/status")
def get_status():
    """A simple status endpoint to confirm the server is running."""
    return {"status": "ok"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from the endpoints module
app.include_router(meta.router)
app.include_router(form.router)
app.include_router(vector.router)

logger.info("✅ FastAPI app initialized with all routers")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 