import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from uuid import UUID

import docker
import numpy as np
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob, Job, JobStatus, TextJob
from miner.logic.job_handler import start_tuning_container, start_tuning_container_diffusion
from miner.logic.advanced_config_generator import AdvancedConfigGenerator
from miner.logic.performance_tracker import PerformanceTracker
from miner.logic.adaptive_trainer import AdaptiveTrainer


logger = get_logger(__name__)


class AdvancedTrainingWorker:
    """
    Advanced training worker with concurrent job limit of 2 and maximum quality techniques.
    Focuses on quality over quantity to achieve top scores.
    """
    
    def __init__(self, max_concurrent_jobs: int = 2):
        logger.info("=" * 80)
        logger.info("STARTING ADVANCED TRAINING WORKER")
        logger.info(f"MAX CONCURRENT JOBS: {max_concurrent_jobs}")
        logger.info("=" * 80)
        
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: Dict[str, Job] = {}
        self.active_jobs: Dict[str, Job] = {}
        self.performance_tracker = PerformanceTracker()
        self.config_generator = AdvancedConfigGenerator()
        
        # Thread pool for job execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        
        # Start worker threads
        self.workers = []
        for i in range(max_concurrent_jobs):
            worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.docker_client = docker.from_env()
        self.shutdown_event = threading.Event()

    def _worker(self, worker_id: int):
        """Individual worker thread for job processing"""
        logger.info(f"Worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get job with timeout to allow shutdown
                job = self.job_queue.get(timeout=1)
                if job is None:
                    break
                
                logger.info(f"Worker {worker_id} processing job {job.job_id}")
                
                # Mark job as active
                self.active_jobs[job.job_id] = job
                
                try:
                    # Generate advanced config for this job
                    advanced_config = self.config_generator.generate_config(job)
                    
                    # Apply advanced training techniques
                    if isinstance(job, TextJob):
                        self._apply_text_optimizations(job, advanced_config)
                        result = start_tuning_container(job)
                    elif isinstance(job, DiffusionJob):
                        self._apply_diffusion_optimizations(job, advanced_config)
                        result = start_tuning_container_diffusion(job)
                    
                    job.status = JobStatus.COMPLETED
                    logger.info(f"Worker {worker_id} completed job {job.job_id}")
                    
                    # Track performance
                    self.performance_tracker.track_job(job.job_id, advanced_config, 0.0, job.__class__.__name__)
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error processing job {job.job_id}: {str(e)}")
                    job.status = JobStatus.FAILED
                    job.error_message = str(e)
                finally:
                    # Remove from active jobs
                    if job.job_id in self.active_jobs:
                        del self.active_jobs[job.job_id]
                    self.job_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {str(e)}")
                time.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")

    def _apply_text_optimizations(self, job: TextJob, config: dict):
        """Apply advanced optimizations for text tasks"""
        logger.info(f"Applying advanced text optimizations for job {job.job_id}")
        
        # Dynamic learning rate based on model size
        model_size_b = self._estimate_model_size(job.model)
        config["learning_rate"] = self._calculate_optimal_lr(model_size_b, job.dataset_type)
        
        # Advanced LoRA configuration
        config["lora_r"] = min(256, max(64, model_size_b * 8))
        config["lora_alpha"] = min(512, max(128, model_size_b * 16))
        config["lora_dropout"] = 0.1
        
        # Dynamic batch size
        config["micro_batch_size"] = min(16, max(4, model_size_b * 2))
        config["gradient_accumulation_steps"] = max(1, 8 // model_size_b)
        
        # Advanced training techniques
        config["gradient_checkpointing"] = model_size_b > 20
        config["early_stopping_patience"] = 3
        config["eval_steps"] = 50
        config["save_steps"] = 100
        
        # Task-specific optimizations
        if hasattr(job.dataset_type, 'task_type'):
            if job.dataset_type.task_type == 'DpoTask':
                config["learning_rate"] *= 1.5
                config["weight_decay"] = 0.01
                config["num_epochs"] = 2
            elif job.dataset_type.task_type == 'GrpoTask':
                config["learning_rate"] *= 1.2
                config["num_epochs"] = 2

    def _apply_diffusion_optimizations(self, job: DiffusionJob, config: dict):
        """Apply advanced optimizations for diffusion tasks"""
        logger.info(f"Applying advanced diffusion optimizations for job {job.job_id}")
        
        # Optimize for H100
        config["train_batch_size"] = 12
        config["learning_rate"] = 0.00003
        config["unet_lr"] = 0.00003
        config["text_encoder_lr"] = 0.00003
        config["network_dim"] = 64
        config["epoch"] = 8
        config["gradient_accumulation_steps"] = 1
        
        # Advanced diffusion techniques
        config["gradient_checkpointing"] = True
        config["mixed_precision"] = "bf16"
        config["xformers"] = True

    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in billions of parameters"""
        model_name_lower = model_name.lower()
        
        if "70b" in model_name_lower or "70b" in model_name_lower:
            return 70.0
        elif "34b" in model_name_lower or "34b" in model_name_lower:
            return 34.0
        elif "13b" in model_name_lower or "13b" in model_name_lower:
            return 13.0
        elif "7b" in model_name_lower or "7b" in model_name_lower:
            return 7.0
        elif "3b" in model_name_lower or "3b" in model_name_lower:
            return 3.0
        elif "1b" in model_name_lower or "1b" in model_name_lower:
            return 1.0
        else:
            return 7.0  # Default assumption

    def _calculate_optimal_lr(self, model_size_b: float, dataset_type) -> float:
        """Calculate optimal learning rate using advanced techniques"""
        base_lr = 0.0002
        
        # Model size adjustments
        if model_size_b > 40:
            base_lr *= 0.5
        elif model_size_b > 20:
            base_lr *= 0.8
        elif model_size_b > 10:
            base_lr *= 1.0
        else:
            base_lr *= 1.2
        
        # Task type adjustments
        if hasattr(dataset_type, 'task_type'):
            if dataset_type.task_type == 'DpoTask':
                base_lr *= 1.5
            elif dataset_type.task_type == 'GrpoTask':
                base_lr *= 1.2
        
        return base_lr

    def enqueue_job(self, job: Job):
        """Enqueue a job for processing"""
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            logger.warning(f"Max concurrent jobs ({self.max_concurrent_jobs}) reached, job {job.job_id} will wait")
        
        self.job_queue.put(job)
        self.job_store[job.job_id] = job
        logger.info(f"Job {job.job_id} enqueued. Active jobs: {len(self.active_jobs)}/{self.max_concurrent_jobs}")

    def get_status(self, job_id: UUID) -> JobStatus:
        """Get status of a job"""
        job = self.job_store.get(str(job_id))
        return job.status if job else JobStatus.NOT_FOUND

    def get_active_jobs_count(self) -> int:
        """Get number of currently active jobs"""
        return len(self.active_jobs)

    def get_queue_size(self) -> int:
        """Get number of jobs waiting in queue"""
        return self.job_queue.qsize()

    def shutdown(self):
        """Shutdown the training worker"""
        logger.info("Shutting down advanced training worker...")
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close docker client
        self.docker_client.close()
        
        logger.info("Advanced training worker shutdown complete") 