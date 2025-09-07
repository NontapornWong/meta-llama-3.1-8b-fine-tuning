# 🚀 Marketing Content Generator

An AI-powered marketing content generator built with fine-tuned Llama 3.1 8B, featuring efficient quantization, LoRA adapters, and production-ready API deployment.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Performance](#performance)
- [Deployment](#deployment)

## 🎯 Overview

This project implements a specialized marketing content generator using Meta's Llama 3.1 8B model, optimized for efficient deployment through 4-bit quantization and fine-tuned with LoRA adapters on marketing-specific examples.

### Key Achievements
- ✅ **68% memory reduction** (16GB → 5GB) through quantization
- ✅ **Fast training** (5.3 minutes for 14 examples)
- ✅ **Quick inference** (5-10 seconds per generation)
- ✅ **Production-ready** API and web interface

## ✨ Features

### 🧠 AI Capabilities
- **Multi-format content generation**: Blog posts, social media, emails, landing pages
- **Professional marketing tone**: Trained on high-quality marketing examples
- **Customizable parameters**: Control length and creativity
- **Fast inference**: Sub-10 second response times

### 🛠 Technical Features
- **Memory efficient**: 4-bit quantization with BitsAndBytes
- **Parameter efficient**: LoRA fine-tuning (1.2% trainable parameters)
- **REST API**: FastAPI with automatic documentation
- **Web interface**: User-friendly Gradio UI
- **Real-time monitoring**: Health checks and performance metrics

### 🎨 User Interface
- **Interactive web app**: No coding required
- **Example prompts**: Quick-start templates
- **Parameter controls**: Adjust length and creativity
- **Copy functionality**: Easy content export
- **API status**: Real-time connection monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (T4 or better)
- 16GB+ GPU memory recommended

### 1. Clone Repository
```bash
git clone https://github.com/your-username/marketing-content-generator.git
cd marketing-content-generator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Models
Place your model files in the following structure:
```
project/
├── my-llama-3.1-8b-local/          # Base Llama model
└── marketing_lora_finetuned/       # Fine-tuned LoRA adapters
```

### 4. Start Services

**Start API Server:**
```bash
python model_apis.py
```
API available at: `http://localhost:8000`

**Start Web Interface:**
```bash
python gradio_app.py
```
UI available at: `http://localhost:7860`

### 5. Generate Content
Visit `http://localhost:7860` and start generating marketing content!

## 📦 Installation

### System Requirements
| Component | Requirement |
|-----------|-------------|
| **GPU** | T4 (16GB VRAM) minimum |
| **RAM** | 16GB+ recommended |
| **Storage** | 20GB for models |
| **Python** | 3.9+ |
| **CUDA** | 11.8+ |

### Dependencies
```bash
# Core ML libraries
pip install torch>=2.0.0 transformers>=4.30.0
pip install peft>=0.4.0 bitsandbytes>=0.39.0 accelerate>=0.20.0

# API and UI
pip install fastapi>=0.104.1 uvicorn>=0.24.0
pip install gradio>=4.0.0

# Utilities
pip install requests datasets
```

### Verify Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎮 Usage

### Web Interface

1. **Open your browser**: `http://localhost:7860`
2. **Enter your prompt**: "Write marketing content about: [your topic]"
3. **Adjust parameters**: Length and creativity sliders
4. **Generate content**: Click the generate button
5. **Copy result**: Use the copy button to save content

### API Usage

**Basic Request:**
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Write marketing content about: AI-powered customer service",
    "max_length": 300,
    "temperature": 0.7
})

result = response.json()
print(result["content"])
```

**Example Prompts:**
- `"Write marketing content about: AI-powered customer service solutions"`
- `"Create a social media campaign for a new productivity app"`
- `"Write an email marketing sequence for Black Friday sales"`
- `"Generate content for a LinkedIn post about remote work trends"`
- `"Create marketing copy for a SaaS landing page"`

## 📚 API Documentation

### Endpoints

#### `POST /generate`
Generate marketing content from a prompt.

**Request Body:**
```json
{
  "prompt": "Your marketing prompt here",
  "max_length": 300,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "content": "Generated marketing content...",
  "prompt": "Your original prompt",
  "generation_time": 8.5
}
```

#### `GET /health`
Check API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_memory_gb": 5.39
}
```

#### `GET /examples`
Get example prompts for inspiration.

**Interactive Documentation:**
Visit `http://localhost:8000/docs` for full API documentation with testing interface.

## 🧠 Model Details

### Architecture
- **Base Model**: Meta Llama 3.1 8B Instruct
- **Quantization**: 4-bit BitsAndBytes (NF4)
- **Fine-tuning**: LoRA adapters
- **Training Data**: 14 marketing examples
- **Context Length**: 512 tokens

### LoRA Configuration
```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Rank
    lora_alpha=16,          # Scaling
    lora_dropout=0.1,       # Regularization
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Training Details
- **Epochs**: 10
- **Batch Size**: 1 (effective: 4 with gradient accumulation)
- **Learning Rate**: 3e-4
- **Training Time**: 5.3 minutes
- **Trainable Parameters**: 1.2% of total

## 📊 Performance

### Benchmarks
| Metric | Value |
|--------|--------|
| **Model Size (Original)** | 16GB |
| **Model Size (Quantized)** | 5GB |
| **Memory Reduction** | 68% |
| **LoRA Adapter Size** | 50MB |
| **Training Time** | 5.3 minutes |
| **Inference Time** | 5-10 seconds |
| **GPU Memory Usage** | 5.39GB |

### Quality Metrics
- ✅ **Professional tone**: Marketing-appropriate language
- ✅ **Coherent structure**: Well-organized content
- ✅ **Domain knowledge**: Marketing-specific terminology
- ✅ **Engaging style**: Action-oriented, persuasive writing

## 🚀 Deployment

### Local Development
```bash
# Start API
python model_apis.py

# Start UI (in another terminal)
python gradio_app.py
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose ports
EXPOSE 8000 7860

# Start services
CMD ["python3", "model_apis.py"]
```

### Cloud Deployment
The application is designed for deployment on:
- **Azure ML**: Managed endpoints with GPU support
- **AWS SageMaker**: Container-based deployment
- **Google Cloud AI Platform**: Custom prediction routines
- **Azure Container Instances**: Direct container deployment

### Environment Variables
```bash
export MODEL_PATH="./my-llama-3.1-8b-local"
export LORA_PATH="./marketing_lora_finetuned"
export API_HOST="0.0.0.0"
export API_PORT="8000"
export UI_PORT="7860"
```

## 🛠 Development

### Project Structure
```
marketing-content-generator/
├── model_apis.py                   # FastAPI service
├── gradio_app.py                   # Web interface
├── requirements.txt                # Dependencies
├── training_data.jsonl            # Training examples
├── models/
    ├── my-llama-3.1-8b-local/     # Base model
    └── marketing_lora_finetuned/   # LoRA adapters
```

## 🙏 Acknowledgments

- **Meta AI** for the Llama 3.1 model
- **Hugging Face** for the transformers library
- **Microsoft** for the BitsAndBytes quantization
- **FastAPI** team for the excellent web framework
- **Gradio** team for the user interface framework

---

