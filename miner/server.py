import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fiber.logging_utils import get_logger

from miner.config import get_worker_config
from miner.endpoints import tuning
from miner.dependencies import get_worker_config_dep


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting advanced miner server...")
    
    # Initialize worker config
    worker_config = get_worker_config()
    logger.info(f"Advanced training worker initialized with {worker_config.trainer.get_max_concurrent_jobs()} concurrent jobs")
    
    yield
    
    logger.info("Shutting down advanced miner server...")
    # Shutdown training worker
    worker_config.trainer.shutdown()


def create_app() -> FastAPI:
    """Create FastAPI application with advanced endpoints"""
    
    app = FastAPI(
        title="Advanced G.O.D Miner",
        description="Advanced miner with maximum quality training techniques",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include standard endpoints (using factory_router)
    standard_router = tuning.factory_router()
    app.include_router(standard_router, prefix="/v1", tags=["standard"])
    
    # Add performance monitoring endpoints
    @app.get("/v1/status")
    async def get_status():
        """Get miner status and performance stats"""
        worker_config = get_worker_config()
        stats = worker_config.trainer.get_performance_stats()
        return {
            "status": "running",
            "active_jobs": worker_config.trainer.get_active_jobs_count(),
            "max_concurrent_jobs": worker_config.trainer.get_max_concurrent_jobs(),
            "performance_stats": stats
        }
    
    @app.get("/v1/performance-report")
    async def get_performance_report():
        """Get detailed performance report"""
        worker_config = get_worker_config()
        stats = worker_config.trainer.get_performance_stats()
        return {
            "performance_report": stats,
            "worker_info": {
                "active_jobs": worker_config.trainer.get_active_jobs_count(),
                "max_concurrent_jobs": worker_config.trainer.get_max_concurrent_jobs(),
                "queue_size": worker_config.trainer.job_queue.qsize()
            }
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 7999))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting advanced miner server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
