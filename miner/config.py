from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from miner.logic.advanced_training_worker import AdvancedTrainingWorker


load_dotenv()


T = TypeVar("T", bound=BaseModel)


@dataclass
class WorkerConfig:
    trainer: AdvancedTrainingWorker


@lru_cache()
def get_worker_config() -> WorkerConfig:
    """Get worker configuration with advanced training worker"""
    # Initialize advanced training worker with concurrent job limit of 2
    advanced_trainer = AdvancedTrainingWorker(max_concurrent_jobs=2)
    return WorkerConfig(trainer=advanced_trainer)
