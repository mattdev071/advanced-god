import queue
import threading
import time
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from uuid import UUID
from concurrent.futures import ThreadPoolExecutor
import yaml
import toml

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import DiffusionJob
from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from core.models.utility_models import TextJob
from miner.logic.job_handler import start_tuning_container
from miner.logic.job_handler import start_tuning_container_diffusion


logger = get_logger(__name__)


@dataclass
class JobResult:
    """Track job performance and results"""
    job_id: str
    config: Dict[str, Any]
    score: float = 0.0
    task_type: str = ""
    model_size: float = 0.0
    training_time: float = 0.0
    timestamp: datetime = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PerformanceTracker:
    """Track and analyze training performance"""
    
    def __init__(self, data_dir: str = "performance_data"):
        self.data_dir = data_dir
        self.results: List[JobResult] = []
        os.makedirs(data_dir, exist_ok=True)
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing performance data"""
        data_file = os.path.join(self.data_dir, "performance_data.json")
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        self.results.append(JobResult(**item))
                logger.info(f"Loaded {len(self.results)} existing performance records")
            except Exception as e:
                logger.error(f"Error loading performance data: {e}")
    
    def _save_data(self):
        """Save performance data to file"""
        data_file = os.path.join(self.data_dir, "performance_data.json")
        try:
            with open(data_file, 'w') as f:
                json.dump([asdict(result) for result in self.results], f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def track_job(self, job_id: str, config: Dict[str, Any], score: float = 0.0, 
                  task_type: str = "", model_size: float = 0.0, training_time: float = 0.0, 
                  success: bool = True, error_message: Optional[str] = None):
        """Track a completed job"""
        result = JobResult(
            job_id=job_id,
            config=config,
            score=score,
            task_type=task_type,
            model_size=model_size,
            training_time=training_time,
            success=success,
            error_message=error_message
        )
        self.results.append(result)
        self._save_data()
        logger.info(f"Tracked job {job_id} with score {score}")
    
    def get_optimal_config(self, task_type: str, model_size: float, 
                          similarity_threshold: float = 0.2) -> Optional[Dict[str, Any]]:
        """Get optimal configuration based on historical performance"""
        # Find similar successful jobs
        similar_jobs = [
            r for r in self.results 
            if (r.success and r.task_type == task_type and 
                abs(r.model_size - model_size) / max(model_size, 1) < similarity_threshold)
        ]
        
        if not similar_jobs:
            return None
        
        # Return config from best performing job
        best_job = max(similar_jobs, key=lambda x: x.score)
        return best_job.config
    
    def get_performance_stats(self, task_type: Optional[str] = None, 
                             days: int = 7) -> Dict[str, Any]:
        """Get performance statistics"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_results = [
            r for r in self.results 
            if r.timestamp > cutoff_date and (task_type is None or r.task_type == task_type)
        ]
        
        if not recent_results:
            return {"total_jobs": 0, "success_rate": 0, "avg_score": 0}
        
        successful_jobs = [r for r in recent_results if r.success]
        return {
            "total_jobs": len(recent_results),
            "success_rate": len(successful_jobs) / len(recent_results),
            "avg_score": sum(r.score for r in successful_jobs) / len(successful_jobs) if successful_jobs else 0,
            "best_score": max(r.score for r in successful_jobs) if successful_jobs else 0
        }


class AdvancedTrainingWorker:
    """Advanced training worker with concurrent job limits and performance tracking"""
    
    def __init__(self, max_concurrent_jobs: int = 2):
        logger.info("=" * 80)
        logger.info(f"STARTING ADVANCED TRAINING WORKER (Max {max_concurrent_jobs} concurrent jobs)")
        logger.info("=" * 80)

        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: Dict[str, Job] = {}
        self.active_jobs: Dict[str, Job] = {}
        self.performance_tracker = PerformanceTracker()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.docker_client = docker.from_env()
        
        # Start worker thread
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _apply_advanced_config(self, job: Job) -> Dict[str, Any]:
        """Apply advanced configuration optimizations"""
        base_config = self._load_base_config(job)
        
        if isinstance(job, TextJob):
            return self._apply_text_optimizations(job, base_config)
        elif isinstance(job, DiffusionJob):
            return self._apply_diffusion_optimizations(job, base_config)
        
        return base_config
    
    def _load_base_config(self, job: Job) -> Dict[str, Any]:
        """Load base configuration based on job type"""
        if isinstance(job, TextJob):
            if "grpo" in job.dataset_type.get("task_type", "").lower():
                config_path = "core/config/base_grpo.yml"
            else:
                config_path = "core/config/base.yml"
        else:  # DiffusionJob
            if "flux" in job.model.lower():
                config_path = "core/config/base_diffusion_flux.toml"
            else:
                config_path = "core/config/base_diffusion_sdxl.toml"
        
        try:
            if config_path.endswith('.yml') or config_path.endswith('.yaml'):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                with open(config_path, 'r') as f:
                    return toml.load(f)
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}
    
    def _apply_text_optimizations(self, job: TextJob, config: dict) -> dict:
        """Apply advanced text training optimizations"""
        # Get optimal config from performance history
        optimal_config = self.performance_tracker.get_optimal_config(
            job.dataset_type.get("task_type", ""), 
            self._estimate_model_size(job.model)
        )
        
        if optimal_config:
            config.update(optimal_config)
        
        # Apply H100-specific optimizations
        config.update({
            "learning_rate": 2e-4,  # Optimized for H100
            "per_device_train_batch_size": 4,  # H100 optimized
            "gradient_accumulation_steps": 4,
            "lora_r": 64,  # Higher rank for better quality
            "lora_alpha": 128,
            "lora_dropout": 0.1,
            "gradient_checkpointing": True,
            "bf16": True,  # H100 native support
            "flash_attention": True,
            "num_epochs": 3,  # Quality over quantity
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
        })
        
        # Task-specific optimizations
        task_type = job.dataset_type.get("task_type", "").lower()
        if "dpo" in task_type:
            config.update({
                "learning_rate": 1e-5,  # Lower LR for DPO
                "num_epochs": 2,
                "beta": 0.1,
                "weight_decay": 0.01,
            })
        elif "grpo" in task_type:
            config.update({
                "learning_rate": 5e-5,  # GRPO specific
                "num_epochs": 2,
                "trl": True,
            })
        
        return config
    
    def _apply_diffusion_optimizations(self, job: DiffusionJob, config: dict) -> dict:
        """Apply advanced diffusion training optimizations"""
        # H100-optimized settings
        config.update({
            "train_batch_size": 4,  # H100 optimized
            "learning_rate": 1e-4,
            "network_dim": 128,  # Higher for better quality
            "network_alpha": 64,
            "num_epochs": 10,  # More epochs for quality
            "mixed_precision": "bf16",  # H100 native
            "xformers": True,
            "cache_latents": True,
            "gradient_checkpointing": True,
            "save_every_n_epochs": 2,
            "validation_prompt": "a photo of a cat",
        })
        
        return config
    
    def _estimate_model_size(self, model_name: str) -> float:
        """Estimate model size in parameters"""
        size_map = {
            "7b": 7e9, "13b": 13e9, "30b": 30e9, "70b": 70e9,
            "1b": 1e9, "3b": 3e9, "6b": 6e9, "8b": 8e9
        }
        
        for size_str, size_val in size_map.items():
            if size_str in model_name.lower():
                return size_val
        
        return 7e9  # Default to 7B
    
    def _worker(self):
        """Main worker loop with concurrent job processing"""
        while True:
            # Check if we can start new jobs
            while len(self.active_jobs) < self.max_concurrent_jobs:
                try:
                    job = self.job_queue.get_nowait()
                    if job is None:
                        return
                    
                    # Start job processing
                    self.active_jobs[job.job_id] = job
                    self.executor.submit(self._process_job, job)
                    
                except queue.Empty:
                    break
            
            time.sleep(1)  # Wait before checking again
    
    def _process_job(self, job: Job):
        """Process a single job with advanced optimizations"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing job {job.job_id} with advanced optimizations")
            
            # Apply advanced configuration
            config = self._apply_advanced_config(job)
            
            # Start training with optimized config
            if isinstance(job, TextJob):
                start_tuning_container(job, config)
            elif isinstance(job, DiffusionJob):
                start_tuning_container_diffusion(job, config)
            
            job.status = JobStatus.COMPLETED
            training_time = time.time() - start_time
            
            # Track performance (score will be updated later)
            self.performance_tracker.track_job(
                job_id=job.job_id,
                config=config,
                task_type=job.dataset_type.get("task_type", ""),
                model_size=self._estimate_model_size(job.model),
                training_time=training_time,
                success=True
            )
            
            logger.info(f"Job {job.job_id} completed successfully in {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing job {job.job_id}: {str(e)}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            
            # Track failed job
            self.performance_tracker.track_job(
                job_id=job.job_id,
                config={},
                task_type=job.dataset_type.get("task_type", ""),
                model_size=self._estimate_model_size(job.model),
                training_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
        
        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            self.job_queue.task_done()

    def enqueue_job(self, job: Job):
        """Enqueue a job for processing"""
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            logger.warning(f"Max concurrent jobs ({self.max_concurrent_jobs}) reached, job {job.job_id} will wait")
        
        self.job_queue.put(job)
        self.job_store[job.job_id] = job
        logger.info(f"Job {job.job_id} enqueued. Active jobs: {len(self.active_jobs)}/{self.max_concurrent_jobs}")

    def get_status(self, job_id: UUID) -> JobStatus:
        """Get job status"""
        job = self.job_store.get(str(job_id))
        return job.status if job else JobStatus.NOT_FOUND
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_tracker.get_performance_stats()
    
    def get_active_jobs_count(self) -> int:
        """Get number of active jobs"""
        return len(self.active_jobs)
    
    def get_max_concurrent_jobs(self) -> int:
        """Get maximum concurrent jobs"""
        return self.max_concurrent_jobs

    def shutdown(self):
        """Shutdown the worker"""
        logger.info("Shutting down advanced training worker...")
        self.executor.shutdown(wait=True)
        self.thread.join()
        self.docker_client.close()


# Keep the original TrainingWorker for backward compatibility
class TrainingWorker(AdvancedTrainingWorker):
    """Backward compatibility wrapper"""
    def __init__(self):
        super().__init__(max_concurrent_jobs=2)  # Default to 2 concurrent jobs
