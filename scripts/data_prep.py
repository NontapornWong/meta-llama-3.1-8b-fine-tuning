from datasets import Dataset
import json

def format_chat_data(json_examples, tokenizer):
    """Convert your JSON examples to training format"""
    
    formatted_texts = []
    
    for example in json_examples:
        messages = example["messages"]
        
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=False
            )
        except:
            formatted_text = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_text += f"<|system|>\n{msg['content']}\n"
                elif msg["role"] == "user":  
                    formatted_text += f"<|user|>\n{msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_text += f"<|assistant|>\n{msg['content']}\n"
            formatted_text += "<|end|>"
        
        formatted_texts.append(formatted_text)
    
    print(f"✅ Formatted {len(formatted_texts)} examples")
    return formatted_texts

def create_training_dataset(formatted_texts, tokenizer, max_length=512):
    """Create the training dataset"""
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    print(f"✅ Tokenized dataset created: {len(tokenized_dataset)} samples")
    return tokenized_dataset

def prepare_your_data_jsonl(jsonl_file_path, tokenizer):
    """Main function to prepare your 14 examples from JSONL"""
    
    print("=== Step 2: Preparing Training Data (JSONL) ===")
    
    try:
        json_examples = []
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                if line.strip(): 
                    json_examples.append(json.loads(line))
        print(f"✅ Loaded {len(json_examples)} examples from {jsonl_file_path}")
    except Exception as e:
        print(f"❌ Could not load JSONL file: {e}")
        print("Make sure your 14 examples are in JSONL format (one JSON per line)")
        return None
    
    formatted_texts = format_chat_data(json_examples, tokenizer)
    
    print("\n=== Sample Formatted Text ===")
    print(formatted_texts[0][:300] + "...")
    
    train_dataset = create_training_dataset(formatted_texts, tokenizer, max_length=512)
    
    return train_dataset