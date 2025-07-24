import asyncio
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import get_config
from pydantic import ValidationError

from core.models.payload_models import MinerTaskOffer, MinerTaskResponse
from core.models.payload_models import TrainRequestText, TrainRequestGrpo, TrainRequestImage
from core.models.utility_models import TaskType, FileFormat
from core.models.tournament_models import TournamentType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config_dep
from miner.logic.job_handler import create_job_text, create_job_diffusion
from miner.logic.performance_tracker import PerformanceTracker
import core.constants as cst


logger = get_logger(__name__)

# Create router
router = APIRouter()

# Global performance tracker
performance_tracker = PerformanceTracker()

# Track current job status
current_job_finish_time = None
active_jobs_count = 0
MAX_CONCURRENT_JOBS = 2


@router.post("/tune-model-text")
async def advanced_tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config_dep),
):
    """Advanced text model tuning with maximum quality focus"""
    global current_job_finish_time, active_jobs_count
    
    logger.info("Starting advanced text model tuning.")
    logger.info(f"Job received: {train_request}")

    # Check concurrent job limit
    if active_jobs_count >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429, 
            detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait for current jobs to complete."
        )

    try:
        # Handle S3 downloads
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(f"Downloaded dataset: {train_request.dataset}")
                train_request.file_format = FileFormat.JSON

        # Create job with advanced configuration
        job = create_job_text(
            job_id=str(train_request.task_id),
            dataset=train_request.dataset,
            model=train_request.model,
            dataset_type=train_request.dataset_type,
            file_format=train_request.file_format,
            expected_repo_name=train_request.expected_repo_name,
        )
        
        logger.info(f"Created advanced job: {job}")
        
        # Update job tracking
        active_jobs_count += 1
        current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
        
        # Enqueue job
        worker_config.trainer.enqueue_job(job)

        return {
            "message": "Advanced training job enqueued with maximum quality settings.", 
            "task_id": job.job_id,
            "active_jobs": active_jobs_count,
            "max_jobs": MAX_CONCURRENT_JOBS
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in advanced text tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced tuning error: {str(e)}")


@router.post("/tune-model-grpo")
async def advanced_tune_model_grpo(
    train_request: TrainRequestGrpo,
    worker_config: WorkerConfig = Depends(get_worker_config_dep),
):
    """Advanced GRPO model tuning with maximum quality focus"""
    global current_job_finish_time, active_jobs_count
    
    logger.info("Starting advanced GRPO model tuning.")
    logger.info(f"Job received: {train_request}")

    # Check concurrent job limit
    if active_jobs_count >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429, 
            detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait for current jobs to complete."
        )

    try:
        # Handle S3 downloads
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(f"Downloaded dataset: {train_request.dataset}")
                train_request.file_format = FileFormat.JSON

        # Create job with advanced configuration
        job = create_job_text(
            job_id=str(train_request.task_id),
            dataset=train_request.dataset,
            model=train_request.model,
            dataset_type=train_request.dataset_type,
            file_format=train_request.file_format,
            expected_repo_name=train_request.expected_repo_name,
        )
        
        logger.info(f"Created advanced GRPO job: {job}")
        
        # Update job tracking
        active_jobs_count += 1
        current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
        
        # Enqueue job
        worker_config.trainer.enqueue_job(job)

        return {
            "message": "Advanced GRPO training job enqueued with maximum quality settings.", 
            "task_id": job.job_id,
            "active_jobs": active_jobs_count,
            "max_jobs": MAX_CONCURRENT_JOBS
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in advanced GRPO tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced GRPO tuning error: {str(e)}")


@router.post("/tune-model-diffusion")
async def advanced_tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config_dep),
):
    """Advanced diffusion model tuning with maximum quality focus"""
    global current_job_finish_time, active_jobs_count
    
    logger.info("Starting advanced diffusion model tuning.")
    logger.info(f"Job received: {train_request}")

    # Check concurrent job limit
    if active_jobs_count >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            status_code=429, 
            detail=f"Maximum concurrent jobs ({MAX_CONCURRENT_JOBS}) reached. Please wait for current jobs to complete."
        )

    try:
        # Download dataset
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
        )
        logger.info(f"Downloaded dataset: {train_request.dataset_zip}")

        # Create job with advanced configuration
        job = create_job_diffusion(
            job_id=str(train_request.task_id),
            dataset_zip=train_request.dataset_zip,
            model=train_request.model,
            model_type=train_request.model_type,
            expected_repo_name=train_request.expected_repo_name,
        )
        
        logger.info(f"Created advanced diffusion job: {job}")
        
        # Update job tracking
        active_jobs_count += 1
        current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
        
        # Enqueue job
        worker_config.trainer.enqueue_job(job)

        return {
            "message": "Advanced diffusion training job enqueued with maximum quality settings.", 
            "task_id": job.job_id,
            "active_jobs": active_jobs_count,
            "max_jobs": MAX_CONCURRENT_JOBS
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in advanced diffusion tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced diffusion tuning error: {str(e)}")


@router.post("/task-offer")
async def advanced_task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config_dep),
) -> MinerTaskResponse:
    """Advanced task offer with maximum quality focus and strategic acceptance"""
    
    try:
        logger.info("Advanced task offer received")
        logger.info(f"Offer details: {request}")
        
        global current_job_finish_time, active_jobs_count
        current_time = datetime.now()
        
        # Priority 1: DPO/GRPO tasks (highest scoring potential)
        if request.task_type in [TaskType.DPOTASK, TaskType.GRPOTASK]:
            if active_jobs_count < MAX_CONCURRENT_JOBS:
                logger.info(f"Accepting high-priority {request.task_type} task")
                return MinerTaskResponse(
                    message=f"Accepting {request.task_type} - optimized for preference learning with maximum quality settings",
                    accepted=True
                )
            else:
                return MinerTaskResponse(
                    message=f"Would accept {request.task_type} but at maximum concurrent jobs ({MAX_CONCURRENT_JOBS})",
                    accepted=False
                )
        
        # Priority 2: Large models (competitive advantage with 8x H100)
        if request.model_params_count and request.model_params_count > 40e9:
            if active_jobs_count < MAX_CONCURRENT_JOBS:
                logger.info(f"Accepting large model task ({request.model_params_count/1e9:.1f}B params)")
                return MinerTaskResponse(
                    message=f"Accepting large model ({request.model_params_count/1e9:.1f}B params) - H100 advantage with maximum quality",
                    accepted=True
                )
            else:
                return MinerTaskResponse(
                    message=f"Would accept large model but at maximum concurrent jobs ({MAX_CONCURRENT_JOBS})",
                    accepted=False
                )
        
        # Priority 3: Tournament participation
        if await _is_tournament_task(request.task_id, config):
            if active_jobs_count < MAX_CONCURRENT_JOBS:
                logger.info("Accepting tournament task - high stakes opportunity")
                return MinerTaskResponse(
                    message="Accepting tournament task - high stakes opportunity with maximum quality",
                    accepted=True
                )
            else:
                return MinerTaskResponse(
                    message=f"Would accept tournament task but at maximum concurrent jobs ({MAX_CONCURRENT_JOBS})",
                    accepted=False
                )
        
        # Priority 4: Proven model families with reasonable time
        if _is_proven_model_family(request.model) and request.hours_to_complete <= 8:
            if active_jobs_count < MAX_CONCURRENT_JOBS:
                logger.info(f"Accepting proven model {request.model}")
                return MinerTaskResponse(
                    message=f"Accepting proven model {request.model} with maximum quality settings",
                    accepted=True
                )
            else:
                return MinerTaskResponse(
                    message=f"Would accept proven model but at maximum concurrent jobs ({MAX_CONCURRENT_JOBS})",
                    accepted=False
                )
        
        # Priority 5: Quick wins for scoring
        if request.hours_to_complete <= 4:
            if active_jobs_count < MAX_CONCURRENT_JOBS:
                logger.info("Accepting quick job for scoring")
                return MinerTaskResponse(
                    message="Accepting quick job for scoring with maximum quality",
                    accepted=True
                )
            else:
                return MinerTaskResponse(
                    message=f"Would accept quick job but at maximum concurrent jobs ({MAX_CONCURRENT_JOBS})",
                    accepted=False
                )
        
        # Reject other tasks
        return MinerTaskResponse(
            message="Declining - focusing on high-value opportunities with maximum quality training",
            accepted=False
        )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in advanced task offer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Advanced task offer error: {str(e)}")


def _is_proven_model_family(model_name: str) -> bool:
    """Check if model is from a proven family"""
    model_lower = model_name.lower()
    proven_families = ["llama", "qwen", "mistral", "gemma", "phi", "falcon"]
    return any(family in model_lower for family in proven_families)


async def _is_tournament_task(task_id: str, config: Config) -> bool:
    """Check if task is part of a tournament"""
    try:
        # This would need to be implemented based on your tournament detection logic
        # For now, return False as placeholder
        return False
    except Exception as e:
        logger.error(f"Error checking tournament status: {e}")
        return False


async def get_advanced_status(
    worker_config: WorkerConfig = Depends(get_worker_config_dep),
) -> dict:
    """Get advanced status information"""
    global current_job_finish_time, active_jobs_count
    
    # Get performance stats
    stats = performance_tracker.get_performance_stats(days=7)
    
    # Get recommendations
    recommendations = performance_tracker.get_recommendations("TextTask", 7.0)  # Example
    
    return {
        "active_jobs": active_jobs_count,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "current_job_finish_time": current_job_finish_time.isoformat() if current_job_finish_time else None,
        "performance_stats": stats,
        "recommendations": recommendations.get("recommendations", []),
        "queue_size": worker_config.trainer.get_queue_size() if hasattr(worker_config.trainer, 'get_queue_size') else 0
    }


async def export_performance_report(
    worker_config: WorkerConfig = Depends(get_worker_config_dep),
) -> dict:
    """Export comprehensive performance report"""
    
    report = performance_tracker.export_report()
    
    return {
        "message": "Performance report exported successfully",
        "report_file": "performance_report.json",
        "summary": report.get("summary", {}),
        "task_types_analyzed": list(report.get("task_type_analysis", {}).keys())
    } 