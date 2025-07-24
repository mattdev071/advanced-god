import os
import yaml
import toml
from typing import Dict, Any
from fiber.logging_utils import get_logger

from core.models.utility_models import TextJob, DiffusionJob, Job
from core.models.utility_models import DpoDatasetType, GrpoDatasetType, InstructTextDatasetType, ChatTemplateDatasetType
from core.models.utility_models import ImageModelType


logger = get_logger(__name__)


class AdvancedConfigGenerator:
    """
    Advanced configuration generator that creates optimized configs for maximum training quality.
    Implements cutting-edge techniques for achieving top scores.
    """
    
    def __init__(self):
        self.base_config_path = "core/config/base.yml"
        self.base_grpo_config_path = "core/config/base_grpo.yml"
        self.base_diffusion_sdxl_path = "core/config/base_diffusion_sdxl.toml"
        self.base_diffusion_flux_path = "core/config/base_diffusion_flux.toml"
        
        # Performance tracking for optimization
        self.performance_history = {}
        
    def generate_config(self, job: Job) -> Dict[str, Any]:
        """Generate advanced configuration for a job"""
        logger.info(f"Generating advanced config for job {job.job_id}")
        
        if isinstance(job, TextJob):
            return self._generate_text_config(job)
        elif isinstance(job, DiffusionJob):
            return self._generate_diffusion_config(job)
        else:
            raise ValueError(f"Unsupported job type: {type(job)}")
    
    def _generate_text_config(self, job: TextJob) -> Dict[str, Any]:
        """Generate advanced text training configuration"""
        
        # Load base config
        if isinstance(job.dataset_type, GrpoDatasetType):
            config_path = self.base_grpo_config_path
        else:
            config_path = self.base_config_path
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply advanced optimizations
        config = self._apply_advanced_text_optimizations(config, job)
        
        # Task-specific optimizations
        if isinstance(job.dataset_type, DpoDatasetType):
            config = self._apply_dpo_optimizations(config, job)
        elif isinstance(job.dataset_type, GrpoDatasetType):
            config = self._apply_grpo_optimizations(config, job)
        elif isinstance(job.dataset_type, (InstructTextDatasetType, ChatTemplateDatasetType)):
            config = self._apply_instruct_optimizations(config, job)
        
        return config
    
    def _generate_diffusion_config(self, job: DiffusionJob) -> Dict[str, Any]:
        """Generate advanced diffusion training configuration"""
        
        # Load base config based on model type
        if job.model_type == ImageModelType.SDXL:
            config_path = self.base_diffusion_sdxl_path
        else:  # FLUX
            config_path = self.base_diffusion_flux_path
            
        with open(config_path, 'r') as f:
            config = toml.load(f)
        
        # Apply advanced diffusion optimizations
        config = self._apply_advanced_diffusion_optimizations(config, job)
        
        return config
    
    def _apply_advanced_text_optimizations(self, config: Dict[str, Any], job: TextJob) -> Dict[str, Any]:
        """Apply advanced optimizations for text training"""
        
        # Model size estimation
        model_size_b = self._estimate_model_size(job.model)
        
        # Advanced LoRA configuration (much higher than defaults)
        config["lora_r"] = min(256, max(64, model_size_b * 8))
        config["lora_alpha"] = min(512, max(128, model_size_b * 16))
        config["lora_dropout"] = 0.1
        config["lora_target_linear"] = True
        
        # Dynamic batch size optimization
        config["micro_batch_size"] = min(16, max(4, model_size_b * 2))
        config["gradient_accumulation_steps"] = max(1, 8 // model_size_b)
        
        # Advanced learning rate calculation
        config["learning_rate"] = self._calculate_optimal_lr(model_size_b, job.dataset_type)
        
        # Advanced training techniques
        config["gradient_checkpointing"] = model_size_b > 20
        config["early_stopping_patience"] = 3
        config["eval_steps"] = 50
        config["save_steps"] = 100
        config["saves_per_epoch"] = 4
        config["evals_per_epoch"] = 4
        
        # Advanced optimizations
        config["bf16"] = True
        config["flash_attention"] = True
        config["xformers_attention"] = False  # H100 native attention is better
        config["tf32"] = False
        
        # Advanced scheduling
        config["lr_scheduler"] = "cosine_with_restarts"
        config["warmup_steps"] = max(10, int(1000 // model_size_b))
        config["weight_decay"] = 0.01
        
        # Advanced evaluation
        config["eval_table_size"] = 10
        config["eval_max_new_tokens"] = 256
        
        return config
    
    def _apply_dpo_optimizations(self, config: Dict[str, Any], job: TextJob) -> Dict[str, Any]:
        """Apply DPO-specific optimizations"""
        logger.info(f"Applying DPO optimizations for job {job.job_id}")
        
        # DPO-specific settings
        config["rl"] = "dpo"
        config["learning_rate"] *= 1.5  # Higher LR for preference learning
        config["weight_decay"] = 0.01
        config["num_epochs"] = 2  # DPO needs more epochs
        config["micro_batch_size"] = min(12, config["micro_batch_size"])
        config["gradient_accumulation_steps"] = max(1, config["gradient_accumulation_steps"] // 2)
        
        # Advanced DPO techniques
        config["beta"] = 0.1  # DPO beta parameter
        config["max_prompt_length"] = 512
        config["max_length"] = 1024
        
        return config
    
    def _apply_grpo_optimizations(self, config: Dict[str, Any], job: TextJob) -> Dict[str, Any]:
        """Apply GRPO-specific optimizations"""
        logger.info(f"Applying GRPO optimizations for job {job.job_id}")
        
        # GRPO-specific settings
        config["rl"] = "grpo"
        config["learning_rate"] *= 1.2
        config["num_epochs"] = 2
        
        # Advanced GRPO techniques
        config["trl"] = {
            "beta": 0.04,
            "max_completion_length": 256,
            "use_vllm": False,
            "num_generations": 4
        }
        
        return config
    
    def _apply_instruct_optimizations(self, config: Dict[str, Any], job: TextJob) -> Dict[str, Any]:
        """Apply instruction tuning optimizations"""
        logger.info(f"Applying instruction tuning optimizations for job {job.job_id}")
        
        # Instruction-specific settings
        config["train_on_inputs"] = False
        config["group_by_length"] = True
        config["sequence_len"] = 2048  # Longer sequences for better instruction following
        
        return config
    
    def _apply_advanced_diffusion_optimizations(self, config: Dict[str, Any], job: DiffusionJob) -> Dict[str, Any]:
        """Apply advanced optimizations for diffusion training"""
        
        # H100-optimized settings
        config["train_batch_size"] = 12
        config["learning_rate"] = 0.00003
        config["unet_lr"] = 0.00003
        config["text_encoder_lr"] = 0.00003
        config["network_dim"] = 64
        config["network_alpha"] = 32
        config["epoch"] = 8
        config["gradient_accumulation_steps"] = 1
        
        # Advanced diffusion techniques
        config["gradient_checkpointing"] = True
        config["mixed_precision"] = "bf16"
        config["xformers"] = True
        config["cache_latents"] = True
        config["cache_latents_to_disk"] = True
        
        # Advanced loss and optimization
        config["loss_type"] = "l2"
        config["huber_c"] = 0.1
        config["huber_schedule"] = "snr"
        config["min_snr_gamma"] = 5
        config["scale_weight_norms"] = 5
        
        # Advanced scheduling
        config["lr_scheduler"] = "constant_with_warmup"
        config["lr_scheduler_args"] = [0.1, 0.1]  # warmup_ratio, min_lr_ratio
        
        # Advanced evaluation
        config["save_every_n_epochs"] = 2
        config["sample_prompts"] = "a photo of a person, a photo of a landscape"
        config["sample_sampler"] = "euler_a"
        
        return config
    
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
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if config_path.endswith('.yml') or config_path.endswith('.yaml'):
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif config_path.endswith('.toml'):
            with open(config_path, 'w') as f:
                toml.dump(config, f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
        
        logger.info(f"Advanced config saved to {config_path}") 