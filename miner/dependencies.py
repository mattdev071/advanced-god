from miner.config import get_worker_config, WorkerConfig


def get_worker_config_dep() -> WorkerConfig:
    """Dependency function to get worker configuration"""
    return get_worker_config()
