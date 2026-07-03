import os
import sys
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="IT Helpdesk Assistant Inference Script")
    parser.add_argument(
        "--question", 
        type=str, 
        default=None, 
        help="A single question/query to ask the assistant. If not provided, starts interactive mode."
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        default="adapters/dpo_lora", 
        help="Path to the saved LoRA adapter directory (default: adapters/dpo_lora)."
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="unsloth/Qwen2.5-1.5B", 
        help="Base model name used to train the adapters (default: unsloth/Qwen2.5-1.5B)."
    )
    parser.add_argument(
        "--force_cpu", 
        action="store_true", 
        help="Force loading on CPU using standard Transformers (bypassing Unsloth/GPU)."
    )
    return parser.parse_args()

def load_model_and_tokenizer(base_model, adapter_path, force_cpu):
    # Check if GPU and Unsloth are available
    gpu_available = torch.cuda.is_available()
    use_unsloth = gpu_available and not force_cpu
    is_adapter_loaded = False
    
    # Check if adapter exists
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        is_adapter_loaded = True

    if use_unsloth:
        try:
            import importlib
            unsloth_module = importlib.import_module("unsloth")
            FastLanguageModel = unsloth_module.FastLanguageModel
            print("GPU and Unsloth detected. Loading model with Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = adapter_path if is_adapter_loaded else base_model,
                max_seq_length = 2048,
                dtype = None,
                load_in_4bit = True
            )
            FastLanguageModel.for_inference(model) # Enable native fast inference
            # Ensure special tokens are added
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
            return model, tokenizer, "unsloth", is_adapter_loaded
        except ImportError:
            print("Unsloth not installed. Falling back to standard Hugging Face...")
    
    # Standard HF Fallback (Supports CPU)
    print("Loading model using standard Hugging Face transformers & peft...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path if is_adapter_loaded else base_model)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    # Register additional special tokens to ensure skip_special_tokens works correctly
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
        
    print(f"Loading base model: {base_model}...")
    device = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None
    )
    
    if is_adapter_loaded:
        print(f"Wrapping model with PEFT adapters from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model_obj, adapter_path)
    else:
        print(f"No adapter found at {adapter_path}. Running base model directly...")
        model = base_model_obj
        
    if device == "cpu":
        model = model.to("cpu")
        
    return model, tokenizer, f"transformers ({device})", is_adapter_loaded

def generate_answer(model, tokenizer, backend, question, is_adapter_loaded):
    system_prompt = "You are a professional IT Helpdesk Assistant. Answer the user's technical questions accurately and professionally."
    
    if is_adapter_loaded:
        # Prompt structure for SFT / DPO adapter models (chat template)
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # Plain text QA prompt for raw pre-trained base models (avoids chat loop/garbage)
        prompt = f"IT Helpdesk Support Q&A\n\nQuestion: {question}\nAnswer:"
    
    # Tokenize input
    inputs = tokenizer([prompt], return_tensors="pt")
    
    # Send inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate parameters
    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    }
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
        
    # Extract only generated response
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    
    # Decode and clean special tokens
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up any leftover special tokens or Chinese formatting junk
    response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
    response = response.replace("\u731e", "") # Remove '猞' character if it somehow leaks
    response = response.replace("user", "").replace("assistant", "")
    
    if not is_adapter_loaded:
        # Prevent base model from continuing to generate new synthesized questions/answers
        for marker in ["\nQuestion:", "\nUser:", "\nIT Support Response:", "\n\n", "Question:"]:
            if marker in response:
                response = response.split(marker)[0]
    
    return response.strip()

def main():
    # Reconfigure stdout to handle UTF-8 print encoding safely on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    
    # Print system information
    print("=" * 60)
    print(" IT Helpdesk AI Assistant - Inference CLI ")
    print("=" * 60)
    
    try:
        model, tokenizer, backend, is_adapter_loaded = load_model_and_tokenizer(
            args.base_model, args.adapter_path, args.force_cpu
        )
        print(f"Model successfully loaded via: {backend}")
        if not is_adapter_loaded:
            print("[NOTICE] Running raw pre-trained base model. Using plain text QA completion format.")
        print("=" * 60)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model/adapters. Details: {e}", file=sys.stderr)
        print("If you haven't trained the adapters yet, start this script with a dummy or base model, or run after training.", file=sys.stderr)
        sys.exit(1)
        
    if args.question:
        # Run a single query
        print(f"\nUser: {args.question}")
        answer = generate_answer(model, tokenizer, backend, args.question, is_adapter_loaded)
        print(f"\nAssistant:\n{answer}\n")
    else:
        # Start interactive shell
        print("Welcome! Type your IT Helpdesk query and press Enter.")
        print("Type 'exit' or 'quit' to close the assistant.\n")
        
        while True:
            try:
                user_input = input("User > ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                if not user_input.strip():
                    continue
                
                print("Processing...")
                answer = generate_answer(model, tokenizer, backend, user_input, is_adapter_loaded)
                print(f"\nAssistant:\n{answer}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error during generation: {e}\n")

if __name__ == "__main__":
    main()
