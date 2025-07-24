import os
from datetime import datetime
from datetime import timedelta

import toml
import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_get_request
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.utility_models import MinerSubmission
from validator.utils.hash_verification import calculate_model_hash

from core.models.payload_models import TrainingRepoResponse

from core.models.payload_models import TrainRequestGrpo
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.models.tournament_models import TournamentType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion
from miner.logic.job_handler import create_job_text


logger = get_logger(__name__)

current_job_finish_time = None


async def tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_grpo(
    train_request: TrainRequestGrpo,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")
    try:
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
        )
        logger.info(train_request.dataset_zip)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip,
        model=train_request.model,
        model_type=train_request.model_type,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def get_latest_model_submission(task_id: str) -> MinerSubmission:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        repo_id = None
        
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                repo_id = config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                repo_id = config_data.get("huggingface_repo_id", None)

        if repo_id is None:
            raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")

        model_hash = calculate_model_hash(repo_id)
        
        return MinerSubmission(repo=repo_id, model_hash=model_hash)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("An offer has come through")
        global current_job_finish_time
        current_time = datetime.now()
        
        # Check if we accept the task type
        if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
            return MinerTaskResponse(
                message=f"This endpoint only accepts text tasks: "
                f"{TaskType.INSTRUCTTEXTTASK}, {TaskType.DPOTASK}, {TaskType.GRPOTASK} and {TaskType.CHATTASK}",
                accepted=False,
            )

        # Strategic task acceptance based on priority
        task_priority = _calculate_task_priority(request)
        logger.info(f"Task priority: {task_priority}")
        
        # Check concurrent job limits
        active_jobs = worker_config.trainer.get_active_jobs_count()
        max_jobs = worker_config.trainer.get_max_concurrent_jobs()
        
        if active_jobs >= max_jobs:
            return MinerTaskResponse(
                message=f"Maximum concurrent jobs ({max_jobs}) reached. Currently processing {active_jobs} jobs.",
                accepted=False,
            )
        
        # Strategic acceptance logic
        if _should_accept_task_strategically(request, task_priority, current_time):
            logger.info(f"Strategically accepting {request.task_type} task with priority {task_priority}")
            return MinerTaskResponse(
                message=f"Strategically accepted {request.task_type} task. Priority: {task_priority}, Active jobs: {active_jobs}/{max_jobs}",
                accepted=True
            )
        else:
            logger.info(f"Strategically rejecting {request.task_type} task with priority {task_priority}")
            return MinerTaskResponse(
                message=f"Strategically rejected {request.task_type} task. Priority: {task_priority}",
                accepted=False
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("An image offer has come through")
        global current_job_finish_time
        current_time = datetime.now()

        if request.task_type != TaskType.IMAGETASK:
            return MinerTaskResponse(message="This endpoint only accepts image tasks", accepted=False)

        if current_job_finish_time is None or current_time + timedelta(hours=1) > current_job_finish_time:
            if request.hours_to_complete < 3:
                logger.info("Accepting the image offer")
                return MinerTaskResponse(message="Yes. I can do image jobs", accepted=True)
            else:
                logger.info("Rejecting offer - too long")
                return MinerTaskResponse(message="I only accept small jobs", accepted=False)
        else:
            return MinerTaskResponse(
                message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/rayonlabs/G.O.D", commit_hash="076e87fc746985e272015322cc91fb3bbbca2f26"
    )


def _calculate_task_priority(request: MinerTaskOffer) -> float:
    """Calculate task priority based on multiple factors"""
    priority = 0.0
    
    # Task type priority (DPO/GRPO first for higher scores)
    task_type_weights = {
        TaskType.DPOTASK: 1.0,
        TaskType.GRPOTASK: 0.9,
        TaskType.INSTRUCTTEXTTASK: 0.7,
        TaskType.CHATTASK: 0.6,
    }
    priority += task_type_weights.get(request.task_type, 0.5) * 10
    
    # Model size priority (larger models preferred)
    model_size = _estimate_model_size(request.model)
    if model_size >= 30e9:  # 30B+
        priority += 5.0
    elif model_size >= 13e9:  # 13B+
        priority += 3.0
    elif model_size >= 7e9:   # 7B+
        priority += 1.0
    
    # Time efficiency (shorter jobs preferred for quick wins)
    if request.hours_to_complete <= 3:
        priority += 3.0
    elif request.hours_to_complete <= 6:
        priority += 1.0
    elif request.hours_to_complete > 12:
        priority -= 2.0
    
    # Proven model families
    if _is_proven_model_family(request.model):
        priority += 2.0
    
    return priority


def _estimate_model_size(model_name: str) -> float:
    """Estimate model size in parameters"""
    size_map = {
        "7b": 7e9, "13b": 13e9, "30b": 30e9, "70b": 70e9,
        "1b": 1e9, "3b": 3e9, "6b": 6e9, "8b": 8e9
    }
    
    for size_str, size_val in size_map.items():
        if size_str in model_name.lower():
            return size_val
    
    return 7e9  # Default to 7B


def _is_proven_model_family(model_name: str) -> bool:
    """Check if model is from a proven family"""
    proven_families = ["llama", "mistral", "qwen", "gemma", "phi"]
    return any(family in model_name.lower() for family in proven_families)


def _should_accept_task_strategically(request: MinerTaskOffer, priority: float, current_time: datetime) -> bool:
    """Determine if task should be accepted based on strategic criteria"""
    global current_job_finish_time
    
    # High priority tasks (DPO/GRPO) are always considered
    if priority >= 12.0:
        return True
    
    # Check if we're currently busy
    if current_job_finish_time and current_time + timedelta(hours=1) > current_job_finish_time:
        # Only accept very high priority tasks when busy
        return priority >= 15.0
    
    # Medium priority tasks need reasonable time constraints
    if priority >= 8.0 and request.hours_to_complete <= 6:
        return True
    
    # Low priority tasks only if we have capacity and they're quick
    if priority >= 5.0 and request.hours_to_complete <= 3:
        return True
    
    return False


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=MinerSubmission,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_grpo/",
        tune_model_grpo,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    return router