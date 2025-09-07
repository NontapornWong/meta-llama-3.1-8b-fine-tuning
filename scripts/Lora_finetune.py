from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
from transformers import TrainingArguments, Trainer
import time

def prepare_model_for_lora(quantized_model):
    """Prepare your quantized model for LoRA training"""
    print("=== Step 3: Preparing Model for LoRA ===")
    
    model = prepare_model_for_kbit_training(quantized_model)
    print("✅ Model prepared for k-bit training")
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing enabled")
    
    return model

def create_lora_config():
    """Create LoRA configuration optimized for small dataset"""
    print("\n=== Creating LoRA Configuration ===")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,                    # Rank - good balance for small dataset
        lora_alpha=16,          # Scaling parameter
        lora_dropout=0.1,       # Dropout for regularization
        target_modules=[
            "q_proj",           # Query projection
            "v_proj",           # Value projection  
            "k_proj",           # Key projection
            "o_proj",           # Output projection
            # Adding these for better performance with small dataset
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        modules_to_save=None,
    )
    
    print("✅ LoRA config created")
    print(f"   - Rank: {lora_config.r}")
    print(f"   - Alpha: {lora_config.lora_alpha}")
    print(f"   - Target modules: {len(lora_config.target_modules)}")
    
    return lora_config

def apply_lora_to_model(prepared_model, lora_config):
    """Apply LoRA adapters to the model"""
    print("\n=== Applying LoRA to Model ===")
    
    try:
        peft_model = get_peft_model(prepared_model, lora_config)
        
        peft_model.print_trainable_parameters()
        
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"\nGPU memory after LoRA: {memory_used:.2f} GB / 15.6 GB")
        
        return peft_model
        
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        return None

def setup_lora_pipeline(quantized_model):
    """Complete LoRA setup pipeline"""
    print("Setting up LoRA for fine-tuning...")
    
    prepared_model = prepare_model_for_lora(quantized_model)
    
    # Create LoRA config
    lora_config = create_lora_config()
    
    # Apply LoRA
    peft_model = apply_lora_to_model(prepared_model, lora_config)
    
    if peft_model is not None:
        print("\nLoRA setup complete")
        print("✅ Model is ready for fine-tuning")
        return peft_model
    else:
        print("\nLoRA setup failed")
        return None

def create_training_arguments():
    """Training arguments optimized for small dataset"""
    print("=== Step 4: Creating Training Configuration ===")
    
    training_args = TrainingArguments(
        # Output settings
        output_dir="./marketing_lora_results",
        overwrite_output_dir=True,
        
        # Training parameters optimized for small dataset
        num_train_epochs=10,                    # More epochs for small dataset
        per_device_train_batch_size=1,          # Small batch 
        gradient_accumulation_steps=4,          # Effective batch size = 4
        
        # Learning rate - higher for small dataset
        learning_rate=3e-4,                     # Slightly higher than usual
        weight_decay=0.01,
        
        # Memory optimizations 
        fp16=True,                              
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        
        # Logging and evaluation
        logging_steps=2,                        # Log every 2 steps (small dataset)
        logging_dir="./logs",
        
        # Saving strategy
        save_strategy="epoch",                  # Save after each epoch
        save_total_limit=3,                     # Keep only 3 checkpoints
        
        # No evaluation (we have only training data)
        eval_strategy="no",
        
        # Optimization
        warmup_steps=5,                         # Small warmup for small dataset
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        
        # Prevent overfitting on small dataset
        max_grad_norm=1.0,
        
        # Other settings
        report_to=None,                         # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    print("✅ Training arguments configured")
    print(f"   - Epochs: {training_args.num_train_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    return training_args

def create_trainer(peft_model, tokenizer, train_dataset, training_args):
    """Create the trainer for fine-tuning"""
    print("\n=== Creating Trainer ===")
    
    try:
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            # No eval dataset since we only have 14 examples
        )
        
        print("✅ Trainer created successfully")
        return trainer
        
    except Exception as e:
        print(f"Error creating trainer: {e}")
        return None

def setup_training(peft_model, tokenizer, train_dataset):
    """Complete training setup"""
    print("Setting up training configuration...")
    
    # Create training arguments
    training_args = create_training_arguments()
    
    # Create trainer
    trainer = create_trainer(peft_model, tokenizer, train_dataset, training_args)
    
    if trainer is not None:
        print("\n✅ Training setup complete")
        
        # Show memory status
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"✅ Current GPU memory: {memory_used:.2f} GB / 15.6 GB")
        
        return trainer
    else:
        print("\nTraining setup failed")
        return None

def start_finetuning(trainer):
    """Start the fine-tuning process"""
    print("Starting Fine-tuning")
    print("=" * 50)
    
    # Clear memory before training
    torch.cuda.empty_cache()
    
    # Record start time
    start_time = time.time()
    
    try:
        print("Training in progress...")
        print("-" * 50)
        
        # Start training
        trainer.train()
        
        # Calculate training time
        end_time = time.time()
        training_time = (end_time - start_time) / 60  # Convert to minutes
        
        print("=" * 50)
        print("Fine-tuning completed successfully!")
        print(f"Training time: {training_time:.1f} minutes")
        
        # Show final memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"Final GPU memory: {memory_used:.2f} GB / 15.6 GB")
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        print("Out of memory error!")
        print("Try reducing batch_size or gradient_accumulation_steps")
        return False
        
    except Exception as e:
        print(f"Training error: {e}")
        return False

def save_model(trainer, peft_model):
    """Save the fine-tuned LoRA adapters"""
    print("\nSaving fine-tuned model...")
    
    try:
        # Save the LoRA adapters
        output_dir = "./marketing_lora_finetuned"
        peft_model.save_pretrained(output_dir)
        
        # Also save the tokenizer
        trainer.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model saved to: {output_dir}")
        print("✅ LoRA adapters and tokenizer saved")
        
        return output_dir
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def complete_finetuning_pipeline(trainer, peft_model):
    """Complete fine-tuning pipeline"""
    print("Starting complete fine-tuning pipeline...")
    
    # Start training
    success = start_finetuning(trainer)
    
    if success:
        # Save the model
        save_path = save_model(trainer, peft_model)
        
        if save_path:
            print("\nFINE-TUNING COMPLETE")
            print(f"✅ Your marketing content generator is ready!")
            print(f"✅ Saved at: {save_path}")
            return True
    return False

# Step 6: Test your fine-tuned marketing content generator

import torch

def test_marketing_generator(peft_model, tokenizer, test_prompts):
    """Test fine-tuned model with marketing prompts"""
    print("Testing Marketing Content Generator")
    print("=" * 60)
    
    peft_model.eval()  # Set to evaluation mode
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt[:50]}...")
        print("-" * 60)
        
        try:
            # Format as chat
            messages = [
                {"role": "system", "content": "You are a marketing expert who creates engaging marketing content."},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(peft_model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = peft_model.generate(
                    **inputs,
                    max_new_tokens=200,      
                    temperature=0.7,         
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            assistant_response = response.split("assistant\n")[-1] if "assistant\n" in response else response
            
            print(f"Generated Content:")
            print(assistant_response[:500] + "..." if len(assistant_response) > 500 else assistant_response)
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"Error generating for prompt {i}: {e}")

def run_marketing_tests(peft_model, tokenizer):
    """Run comprehensive tests of your marketing generator"""
    
    # Test prompts covering different marketing scenarios
    test_prompts = [
        "Write marketing content about: The Future of AI in Digital Marketing",
        "Create a social media campaign for a new productivity app",
        "Write an email marketing sequence for Black Friday sales",
        "Generate content for a LinkedIn post about remote work trends",
        "Create marketing copy for a SaaS landing page"
    ]
    
    print("🚀 Running Marketing Generator Tests")
    print(f"Testing {len(test_prompts)} different marketing scenarios...")
    
    test_marketing_generator(peft_model, tokenizer, test_prompts)
    
    print("\nTesting Complete")
    print("Evaluate the results:")
    print("   - Does it sound like marketing content?")
    print("   - Is the tone professional and engaging?")
    print("   - Does it follow your training style?")

def quick_test(peft_model, tokenizer, custom_prompt=None):
    """Quick single test of your model"""
    
    if custom_prompt is None:
        custom_prompt = "Write marketing content about: How AI is Revolutionizing Customer Experience"
    
    print(f"Quick Test: {custom_prompt}")
    print("-" * 50)
    
    test_marketing_generator(peft_model, tokenizer, [custom_prompt])