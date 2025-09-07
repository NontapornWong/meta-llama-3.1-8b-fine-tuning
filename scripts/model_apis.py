from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import uvicorn
from contextlib import asynccontextmanager
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None

class MarketingRequest(BaseModel):
    """Request model for marketing content generation"""
    prompt: str
    max_length: int = 300
    temperature: float = 0.7

class MarketingResponse(BaseModel):
    """Response model for marketing content"""
    content: str
    prompt: str
    generation_time: float

class ModelConfig:
    """Configuration for your model paths"""
    BASE_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  
    LORA_ADAPTER_PATH = "./marketing_lora_finetuned" 

def load_marketing_model():
    """Load the quantized model with LoRA adapters"""
    global model, tokenizer
    
    logger.info("Loading Marketing Content Generator...")
    logger.info(f"Base model: {ModelConfig.BASE_MODEL_PATH}")
    logger.info(f"LoRA adapters: {ModelConfig.LORA_ADAPTER_PATH}")

    try:
        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load base model with quantization
        logger.info("Loading base model with 4-bit quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            ModelConfig.BASE_MODEL_PATH,
            quantization_config=bnb_config,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters
        logger.info("Loading fine-tuned LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, ModelConfig.LORA_ADAPTER_PATH)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(ModelConfig.LORA_ADAPTER_PATH)
        
        # Set to evaluation mode
        model.eval()
        
        memory_used = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"GPU memory usage: {memory_used:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
    
def generate_marketing_content(prompt: str, max_length: int = 300, temperature: float = 0.7):
    """Generate marketing content from a prompt"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Format as chat
    messages = [
        {"role": "system", "content": "You are a marketing expert who creates engaging marketing content."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        start_time = time.time()
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant\n" in response:
            assistant_response = response.split("assistant\n")[-1]
        else:
            assistant_response = response
        
        generation_time = time.time() - start_time
        
        return assistant_response.strip(), generation_time
        
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Marketing Generator API...")
    success = load_marketing_model()
    if not success:
        logger.error("Failed to load model. Exiting.")
        exit(1)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Marketing Generator API...")

# Create FastAPI app
app = FastAPI(
    title="Marketing Content Generator API",
    description="Generate engaging marketing content using fine-tuned Llama 3.1 8B",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Marketing Content Generator API is running!",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_gb": round(gpu_memory, 2)
    }

@app.post("/generate", response_model=MarketingResponse)
async def generate_content(request: MarketingRequest):
    """Generate marketing content"""
    
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if request.max_length > 500:
        raise HTTPException(status_code=400, detail="max_length cannot exceed 500")
    
    if not 0.1 <= request.temperature <= 2.0:
        raise HTTPException(status_code=400, detail="temperature must be between 0.1 and 2.0")
    
    logger.info(f"Generating content for prompt: {request.prompt[:50]}...")
    
    try:
        content, generation_time = generate_marketing_content(
            request.prompt, 
            request.max_length, 
            request.temperature
        )
        
        logger.info(f"Generated content in {generation_time:.2f}s")
        
        return MarketingResponse(
            content=content,
            prompt=request.prompt,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Content generation failed")

@app.get("/examples")
async def get_examples():
    """Get example prompts"""
    return {
        "examples": [
            "Write marketing content about: AI-powered customer service solutions",
            "Create a social media campaign for a new productivity app",
            "Write an email marketing sequence for Black Friday sales",
            "Generate content for a LinkedIn post about remote work trends",
            "Create marketing copy for a SaaS landing page"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "model_apis:app", 
        host="0.0.0.0",
        port=8000,
        reload=False
    )