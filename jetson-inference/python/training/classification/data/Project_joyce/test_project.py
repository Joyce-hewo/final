from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def generate_recommendations():
    try:
        # Configuration
        access_token = "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"  # Consider using environment variables
        model_name = "NousResearch/Llama-2-7b-hf"
        
        # Device setup
        device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        
        # Model loading - removed redundant device_map since we're handling it in pipeline
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            torch_dtype=torch.float16  # FP16 for memory efficiency
        )

        # Create pipeline with proper device mapping
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,  # Simplified device assignment
            torch_dtype=torch.float16  # Ensure FP16 throughout
        )
        
        # Generate text with more controlled parameters
        output = generator(
            'Hi! Who are you?',
            do_sample=True,
            max_new_tokens=50,  # Reduced for Jetson stability
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id  # Ensure proper termination
        )
        
        # Print and return the result
        result = output[0]['generated_text']
        print("Generated Text:", result)
        return result
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Test the function
if __name__ == "__main__":
    generate_recommendations()