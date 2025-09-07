import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import gc

def setup_quantization_config():
    print("=== Setting up Quantization Configuration ===")
    
    bnb_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,      
        bnb_4bit_quant_type="nf4",           
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_storage=torch.uint8   
    )
    
    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,              
        llm_int8_has_fp16_weight=False      
    )
    
    print("Quantization configs ready")
    return bnb_config_4bit, bnb_config_8bit

def load_quantized_model(model_path, use_4bit=True):
    """Load your local model with quantization"""
    print(f"\n=== Loading Model from: {model_path} ===")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    config_4bit, config_8bit = setup_quantization_config()
    config = config_4bit if use_4bit else config_8bit
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        quant_type = "4-bit" if use_4bit else "8-bit"
        print(f"Loading model with {quant_type} quantization...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=config,
            dtype=torch.float16,
            device_map="auto",              
            trust_remote_code=True,
            low_cpu_mem_usage=True       
        )
        
        # Check memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"\n✅ Model loaded successfully!")
        print(f"GPU memory used: {memory_used:.2f} GB / 15.6 GB")
        print(f"Quantization: {quant_type}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
def test_quantized_model(model, tokenizer):
    """Quick test of the quantized model"""
    print("\n=== Testing Quantized Model ===")
    
    test_prompt = "Hello! Can you tell me what is machine learning?"
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n✅ Model test successful!")
        print(f"Input: {test_prompt}")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
def verify_model_state(model):
    """Check your loaded model is ready"""
    print("\n=== Model Verification ===")
    print(f"Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Check if model is quantized
    if hasattr(model, 'hf_quantizer'):
        print("✅ Model is quantized with BitsAndBytes")
    else:
        print("⚠️ Model quantization not detected")
    
    return True
