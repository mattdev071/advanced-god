import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from fiber.logging_utils import get_logger

import numpy as np


logger = get_logger(__name__)


@dataclass
class JobResult:
    """Job result data for performance tracking"""
    job_id: str
    config: Dict[str, Any]
    score: float
    task_type: str
    model_size: float
    training_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None


class PerformanceTracker:
    """
    Advanced performance tracker for monitoring and optimizing training performance.
    Tracks job results and provides insights for optimization.
    """
    
    def __init__(self, data_dir: str = "performance_data"):
        self.data_dir = data_dir
        self.results: List[JobResult] = []
        self.performance_cache: Dict[str, Any] = {}
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing results
        self._load_results()
        
    def track_job(self, job_id: str, config: Dict[str, Any], score: float, task_type: str, 
                  model_size: float = 0.0, training_time: float = 0.0, success: bool = True, 
                  error_message: Optional[str] = None):
        """Track a job result"""
        
        result = JobResult(
            job_id=job_id,
            config=config,
            score=score,
            task_type=task_type,
            model_size=model_size,
            training_time=training_time,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        self.results.append(result)
        self._save_results()
        
        logger.info(f"Tracked job {job_id}: score={score}, task_type={task_type}, success={success}")
    
    def get_optimal_config(self, task_type: str, model_size: float, 
                          similarity_threshold: float = 0.2) -> Optional[Dict[str, Any]]:
        """Get optimal configuration for similar tasks"""
        
        # Filter similar jobs
        similar_jobs = [
            r for r in self.results
            if r.task_type == task_type and 
               r.success and
               abs(r.model_size - model_size) / max(model_size, 1.0) < similarity_threshold
        ]
        
        if not similar_jobs:
            logger.info(f"No similar successful jobs found for {task_type} with model size {model_size}B")
            return None
        
        # Sort by score (descending)
        similar_jobs.sort(key=lambda x: x.score, reverse=True)
        
        # Return config from best performing job
        best_job = similar_jobs[0]
        logger.info(f"Found optimal config from job {best_job.job_id} with score {best_job.score}")
        
        return best_job.config
    
    def get_performance_stats(self, task_type: Optional[str] = None, 
                            days: int = 7) -> Dict[str, Any]:
        """Get performance statistics"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter results
        filtered_results = [
            r for r in self.results
            if r.timestamp >= cutoff_date and
               (task_type is None or r.task_type == task_type) and
               r.success
        ]
        
        if not filtered_results:
            return {
                "total_jobs": 0,
                "success_rate": 0.0,
                "avg_score": 0.0,
                "best_score": 0.0,
                "avg_training_time": 0.0
            }
        
        scores = [r.score for r in filtered_results]
        training_times = [r.training_time for r in filtered_results if r.training_time > 0]
        
        stats = {
            "total_jobs": len(filtered_results),
            "success_rate": len(filtered_results) / len([r for r in self.results if r.timestamp >= cutoff_date]),
            "avg_score": np.mean(scores),
            "best_score": max(scores),
            "score_std": np.std(scores),
            "avg_training_time": np.mean(training_times) if training_times else 0.0,
            "task_type": task_type,
            "period_days": days
        }
        
        return stats
    
    def get_top_performers(self, task_type: Optional[str] = None, 
                          limit: int = 10) -> List[JobResult]:
        """Get top performing jobs"""
        
        filtered_results = [
            r for r in self.results
            if r.success and (task_type is None or r.task_type == task_type)
        ]
        
        # Sort by score (descending)
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_results[:limit]
    
    def analyze_config_effectiveness(self, task_type: str) -> Dict[str, Any]:
        """Analyze configuration effectiveness for a task type"""
        
        task_results = [r for r in self.results if r.task_type == task_type and r.success]
        
        if not task_results:
            return {"error": f"No successful results found for {task_type}"}
        
        # Analyze learning rate effectiveness
        lr_analysis = self._analyze_learning_rates(task_results)
        
        # Analyze LoRA configuration effectiveness
        lora_analysis = self._analyze_lora_configs(task_results)
        
        # Analyze batch size effectiveness
        batch_analysis = self._analyze_batch_sizes(task_results)
        
        return {
            "task_type": task_type,
            "total_jobs": len(task_results),
            "learning_rate_analysis": lr_analysis,
            "lora_analysis": lora_analysis,
            "batch_size_analysis": batch_analysis
        }
    
    def _analyze_learning_rates(self, results: List[JobResult]) -> Dict[str, Any]:
        """Analyze learning rate effectiveness"""
        
        lr_scores = {}
        for result in results:
            lr = result.config.get("learning_rate", 0.0)
            if lr > 0:
                if lr not in lr_scores:
                    lr_scores[lr] = []
                lr_scores[lr].append(result.score)
        
        if not lr_scores:
            return {"error": "No learning rate data found"}
        
        analysis = {}
        for lr, scores in lr_scores.items():
            analysis[f"lr_{lr}"] = {
                "count": len(scores),
                "avg_score": np.mean(scores),
                "best_score": max(scores),
                "std_score": np.std(scores)
            }
        
        return analysis
    
    def _analyze_lora_configs(self, results: List[JobResult]) -> Dict[str, Any]:
        """Analyze LoRA configuration effectiveness"""
        
        lora_scores = {}
        for result in results:
            lora_r = result.config.get("lora_r", 0)
            lora_alpha = result.config.get("lora_alpha", 0)
            
            if lora_r > 0 and lora_alpha > 0:
                config_key = f"r{lora_r}_alpha{lora_alpha}"
                if config_key not in lora_scores:
                    lora_scores[config_key] = []
                lora_scores[config_key].append(result.score)
        
        if not lora_scores:
            return {"error": "No LoRA configuration data found"}
        
        analysis = {}
        for config, scores in lora_scores.items():
            analysis[config] = {
                "count": len(scores),
                "avg_score": np.mean(scores),
                "best_score": max(scores),
                "std_score": np.std(scores)
            }
        
        return analysis
    
    def _analyze_batch_sizes(self, results: List[JobResult]) -> Dict[str, Any]:
        """Analyze batch size effectiveness"""
        
        batch_scores = {}
        for result in results:
            batch_size = result.config.get("micro_batch_size", 0)
            if batch_size > 0:
                if batch_size not in batch_scores:
                    batch_scores[batch_size] = []
                batch_scores[batch_size].append(result.score)
        
        if not batch_scores:
            return {"error": "No batch size data found"}
        
        analysis = {}
        for batch_size, scores in batch_scores.items():
            analysis[f"batch_{batch_size}"] = {
                "count": len(scores),
                "avg_score": np.mean(scores),
                "best_score": max(scores),
                "std_score": np.std(scores)
            }
        
        return analysis
    
    def get_recommendations(self, task_type: str, model_size: float) -> Dict[str, Any]:
        """Get training recommendations based on historical data"""
        
        # Get optimal config
        optimal_config = self.get_optimal_config(task_type, model_size)
        
        # Get performance stats
        stats = self.get_performance_stats(task_type, days=30)
        
        # Get top performers
        top_performers = self.get_top_performers(task_type, limit=5)
        
        # Analyze config effectiveness
        config_analysis = self.analyze_config_effectiveness(task_type)
        
        recommendations = {
            "task_type": task_type,
            "model_size": model_size,
            "optimal_config": optimal_config,
            "performance_stats": stats,
            "top_performers": [
                {
                    "job_id": r.job_id,
                    "score": r.score,
                    "model_size": r.model_size,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in top_performers
            ],
            "config_analysis": config_analysis,
            "recommendations": self._generate_recommendations(stats, config_analysis)
        }
        
        return recommendations
    
    def _generate_recommendations(self, stats: Dict[str, Any], 
                                config_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on analysis"""
        
        recommendations = []
        
        # Performance-based recommendations
        if stats["avg_score"] < 2.0:
            recommendations.append("Consider increasing LoRA rank for better performance")
        
        if stats["success_rate"] < 0.8:
            recommendations.append("Focus on improving job success rate - check for common failure patterns")
        
        # Learning rate recommendations
        if "learning_rate_analysis" in config_analysis:
            lr_analysis = config_analysis["learning_rate_analysis"]
            if "error" not in lr_analysis:
                best_lr = max(lr_analysis.items(), key=lambda x: x[1]["avg_score"])
                recommendations.append(f"Best performing learning rate: {best_lr[0]} (avg score: {best_lr[1]['avg_score']:.3f})")
        
        # LoRA recommendations
        if "lora_analysis" in config_analysis:
            lora_analysis = config_analysis["lora_analysis"]
            if "error" not in lora_analysis:
                best_lora = max(lora_analysis.items(), key=lambda x: x[1]["avg_score"])
                recommendations.append(f"Best performing LoRA config: {best_lora[0]} (avg score: {best_lora[1]['avg_score']:.3f})")
        
        return recommendations
    
    def _save_results(self):
        """Save results to file"""
        results_file = os.path.join(self.data_dir, "job_results.json")
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            result_dict["timestamp"] = result.timestamp.isoformat()
            serializable_results.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} job results to {results_file}")
    
    def _load_results(self):
        """Load results from file"""
        results_file = os.path.join(self.data_dir, "job_results.json")
        
        if not os.path.exists(results_file):
            logger.info("No existing results file found")
            return
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            self.results = []
            for item in data:
                # Convert timestamp back to datetime
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                self.results.append(JobResult(**item))
            
            logger.info(f"Loaded {len(self.results)} job results from {results_file}")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            self.results = []
    
    def export_report(self, output_file: str = "performance_report.json"):
        """Export comprehensive performance report"""
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_jobs": len(self.results),
                "successful_jobs": len([r for r in self.results if r.success]),
                "success_rate": len([r for r in self.results if r.success]) / len(self.results) if self.results else 0.0
            },
            "task_type_analysis": {},
            "top_performers": {},
            "recommendations": {}
        }
        
        # Analyze each task type
        task_types = set(r.task_type for r in self.results)
        for task_type in task_types:
            report["task_type_analysis"][task_type] = self.get_performance_stats(task_type, days=30)
            report["top_performers"][task_type] = [
                {
                    "job_id": r.job_id,
                    "score": r.score,
                    "model_size": r.model_size,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.get_top_performers(task_type, limit=10)
            ]
            report["recommendations"][task_type] = self.analyze_config_effectiveness(task_type)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report exported to {output_file}")
        return report 