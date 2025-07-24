# Advanced G.O.D Miner

## Overview

This is an enhanced version of the G.O.D miner with advanced training techniques optimized for achieving top scores with 8x H100 GPUs.

## Key Features

### 🚀 **Advanced Training Worker**
- **2 concurrent jobs maximum** for quality over quantity
- **Performance tracking** and learning from past results
- **H100-optimized configurations** with BF16, Flash Attention, and native optimizations
- **Task-specific optimizations** for DPO, GRPO, Instruct, and Diffusion tasks

### 🎯 **Strategic Task Acceptance**
- **Priority-based selection** prioritizing DPO/GRPO tasks (higher scoring potential)
- **Model size consideration** (larger models preferred)
- **Proven model families** (Llama, Mistral, Qwen, Gemma, Phi)
- **Time efficiency** (shorter jobs for quick wins)

### 📊 **Performance Monitoring**
- **Real-time statistics** via `/v1/status` endpoint
- **Performance reports** via `/v1/performance-report` endpoint
- **Historical data tracking** for continuous improvement

## Advanced Optimizations

### **Text Training (DPO/GRPO/Instruct)**
```yaml
# H100-optimized settings
learning_rate: 2e-4
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
lora_r: 64
lora_alpha: 128
lora_dropout: 0.1
gradient_checkpointing: true
bf16: true
flash_attention: true
num_epochs: 3
```

### **Diffusion Training**
```toml
# H100-optimized settings
train_batch_size = 4
learning_rate = 1e-4
network_dim = 128
network_alpha = 64
num_epochs = 10
mixed_precision = "bf16"
xformers = true
cache_latents = true
gradient_checkpointing = true
```

### **Task-Specific Optimizations**
- **DPO**: Lower learning rate (1e-5), 2 epochs, beta=0.1
- **GRPO**: Medium learning rate (5e-5), 2 epochs, TRL enabled
- **Instruct**: Standard optimizations with quality focus

## API Endpoints

### **Training Endpoints**
- `POST /v1/start_training/` - Advanced text model tuning
- `POST /v1/start_training_grpo/` - Advanced GRPO model tuning
- `POST /v1/start_training_image/` - Advanced diffusion model tuning

### **Task Management**
- `POST /v1/task_offer/` - Strategic task acceptance
- `POST /v1/task_offer_image/` - Image task acceptance

### **Monitoring**
- `GET /v1/status` - Real-time status and performance stats
- `GET /v1/performance-report` - Detailed performance report

## Strategic Task Acceptance Logic

### **Priority Calculation**
1. **Task Type Priority**:
   - DPO: 10.0 points
   - GRPO: 9.0 points
   - Instruct: 7.0 points
   - Chat: 6.0 points

2. **Model Size Bonus**:
   - 30B+: +5.0 points
   - 13B+: +3.0 points
   - 7B+: +1.0 points

3. **Time Efficiency**:
   - ≤3 hours: +3.0 points
   - ≤6 hours: +1.0 point
   - >12 hours: -2.0 points

4. **Proven Model Family**: +2.0 points

### **Acceptance Criteria**
- **High Priority (≥12.0)**: Always accepted
- **Medium Priority (≥8.0)**: Accepted if ≤6 hours
- **Low Priority (≥5.0)**: Accepted if ≤3 hours and available capacity

## Running the Advanced Miner

### **Start the Server**
```bash
python miner/server.py
```

### **Check Status**
```bash
curl http://localhost:7999/v1/status
```

### **Get Performance Report**
```bash
curl http://localhost:7999/v1/performance-report
```

## Performance Data

The system automatically tracks:
- Job configurations and results
- Training times and success rates
- Model performance scores
- Task type effectiveness

Data is stored in `performance_data/performance_data.json` and used to optimize future training configurations.

## Competitive Advantages

1. **Quality over Quantity**: 2 concurrent jobs ensure maximum quality per job
2. **Strategic Task Selection**: Prioritizes high-scoring opportunities
3. **H100 Optimizations**: Leverages full H100 capabilities
4. **Learning System**: Continuously improves based on performance data
5. **Proven Configurations**: Uses optimized settings for each task type

## Configuration Files

The system uses existing configuration files with advanced optimizations:
- `core/config/base.yml` - Base text training config
- `core/config/base_grpo.yml` - Base GRPO config
- `core/config/base_diffusion_sdxl.toml` - SDXL diffusion config
- `core/config/base_diffusion_flux.toml` - Flux diffusion config

Advanced optimizations are applied on top of these base configurations.

## Monitoring and Debugging

### **Logs**
- All training operations are logged with detailed information
- Performance tracking logs show optimization decisions
- Strategic task acceptance logs show priority calculations

### **Performance Data**
- Historical job results stored in JSON format
- Configurable retention period (default: 7 days)
- Exportable performance reports

This advanced miner is designed to maximize your chances of achieving top scores in the G.O.D network by focusing on quality, strategic task selection, and continuous learning from performance data. 