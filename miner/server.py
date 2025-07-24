import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fiber.logging_utils import get_logger

from miner.config import get_worker_config
from miner.endpoints import tuning
from miner.endpoints import advanced_tuning
from miner.dependencies import get_worker_config_dep


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting advanced miner server...")
    
    # Initialize advanced worker config
    worker_config = get_worker_config()
    logger.info(f"Advanced training worker initialized with {worker_config.trainer.max_concurrent_jobs} concurrent jobs")
    
    yield
    
    logger.info("Shutting down advanced miner server...")
    # Shutdown advanced training worker
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
    
    # Include standard endpoints
    app.include_router(tuning.router, prefix="/v1", tags=["standard"])
    
    # Include advanced endpoints
    app.include_router(advanced_tuning.router, prefix="/v1/advanced", tags=["advanced"])
    
    # Add advanced status endpoint
    @app.get("/v1/advanced/status")
    async def get_advanced_status():
        """Get advanced miner status"""
        return await advanced_tuning.get_advanced_status()
    
    # Add performance report endpoint
    @app.get("/v1/advanced/performance-report")
    async def export_performance_report():
        """Export performance report"""
        return await advanced_tuning.export_performance_report()
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 7999))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting advanced miner server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
