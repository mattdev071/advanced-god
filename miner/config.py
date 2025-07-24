from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from miner.logic.training_worker import TrainingWorker


load_dotenv()


T = TypeVar("T", bound=BaseModel)


@dataclass
class WorkerConfig:
    trainer: TrainingWorker


@lru_cache()
def get_worker_config() -> WorkerConfig:
    """Get worker configuration with training worker"""
    # Initialize training worker
    trainer = TrainingWorker()
    return WorkerConfig(trainer=trainer)
