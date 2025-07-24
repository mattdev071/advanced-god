#!/usr/bin/env python3
"""
Simple test script for the Advanced G.O.D Miner
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from miner.config import get_worker_config, WorkerConfig
        print("✅ miner.config imported successfully")
        
        from miner.logic.training_worker import TrainingWorker, AdvancedTrainingWorker, PerformanceTracker
        print("✅ miner.logic.training_worker imported successfully")
        
        from miner.endpoints.tuning import factory_router
        print("✅ miner.endpoints.tuning imported successfully")
        
        from miner.server import create_app
        print("✅ miner.server imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_worker_initialization():
    """Test that the training worker initializes correctly"""
    print("\nTesting worker initialization...")
    
    try:
        from miner.config import get_worker_config
        
        worker_config = get_worker_config()
        print(f"✅ Worker config created successfully")
        print(f"   - Trainer type: {type(worker_config.trainer).__name__}")
        print(f"   - Max concurrent jobs: {worker_config.trainer.get_max_concurrent_jobs()}")
        print(f"   - Active jobs: {worker_config.trainer.get_active_jobs_count()}")
        
        return True
    except Exception as e:
        print(f"❌ Worker initialization error: {e}")
        return False

def test_performance_tracker():
    """Test the performance tracker functionality"""
    print("\nTesting performance tracker...")
    
    try:
        from miner.logic.training_worker import PerformanceTracker
        
        tracker = PerformanceTracker("test_performance_data")
        print("✅ Performance tracker created successfully")
        
        # Test tracking a job
        tracker.track_job(
            job_id="test_job_1",
            config={"learning_rate": 2e-4, "batch_size": 4},
            score=0.85,
            task_type="InstructTextTask",
            model_size=7e9,
            training_time=3600.0,
            success=True
        )
        print("✅ Job tracking test successful")
        
        # Test getting stats
        stats = tracker.get_performance_stats()
        print(f"✅ Performance stats retrieved: {stats}")
        
        # Clean up test data
        import shutil
        if os.path.exists("test_performance_data"):
            shutil.rmtree("test_performance_data")
        
        return True
    except Exception as e:
        print(f"❌ Performance tracker error: {e}")
        return False

def test_priority_calculation():
    """Test the strategic task acceptance logic"""
    print("\nTesting priority calculation...")
    
    try:
        from miner.endpoints.tuning import _calculate_task_priority, _estimate_model_size, _is_proven_model_family
        from core.models.payload_models import MinerTaskOffer
        from core.models.utility_models import TaskType
        
        # Test model size estimation
        assert _estimate_model_size("meta-llama/Llama-2-7b-hf") == 7e9
        assert _estimate_model_size("meta-llama/Llama-2-13b-hf") == 13e9
        print("✅ Model size estimation working")
        
        # Test proven model family detection
        assert _is_proven_model_family("meta-llama/Llama-2-7b-hf") == True
        assert _is_proven_model_family("microsoft/DialoGPT-medium") == False
        print("✅ Proven model family detection working")
        
        # Test priority calculation
        request = MinerTaskOffer(
            task_type=TaskType.DPOTASK,
            model="meta-llama/Llama-2-13b-hf",
            hours_to_complete=4
        )
        
        priority = _calculate_task_priority(request)
        print(f"✅ Priority calculation working: {priority}")
        
        return True
    except Exception as e:
        print(f"❌ Priority calculation error: {e}")
        return False

def test_server_creation():
    """Test that the FastAPI server can be created"""
    print("\nTesting server creation...")
    
    try:
        from miner.server import create_app
        
        app = create_app()
        print("✅ FastAPI app created successfully")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
        
        # Test that routes are registered
        routes = [route.path for route in app.routes]
        print(f"   - Routes: {len(routes)} routes registered")
        
        return True
    except Exception as e:
        print(f"❌ Server creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Advanced G.O.D Miner System")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_worker_initialization,
        test_performance_tracker,
        test_priority_calculation,
        test_server_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Advanced G.O.D Miner is ready to use.")
        print("\n🚀 To start the miner:")
        print("   python miner/server.py")
        print("\n📊 To check status:")
        print("   curl http://localhost:7999/v1/status")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 