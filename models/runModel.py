import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, infer_auto_device_map

def load_model(model_name, quantization='none'):
    """
    Load the model and tokenizer with the specified quantization level and device mapping.
    
    Args:
        model_name (str): Name of the model to load.
        quantization (str): Quantization level, options are 'none', '4bit', or '8bit'.
        
    Returns:
        tuple: Loaded model and tokenizer.
    """
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_name)
    device_map = infer_auto_device_map(model, max_memory={0: "24GB", 1: "24GB", 2: "24GB", 3: "24GB"})
    print(f"Using device map: {device_map}")

    if quantization == '8bit':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True
        )
    elif quantization == '4bit':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_4bit=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    return model, tokenizer

def generate_sparql(model, tokenizer, input_text):
    """
    Generate a SPARQL query from a given input prompt using the provided model and tokenizer.
    
    Args:
        model (transformers.PreTrainedModel): Pre-trained model for text generation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the model.
        input_text (str): The input prompt to generate a SPARQL query.
    
    Returns:
        tuple: Generated SPARQL query (str) and time taken (float).
    """
    print("Tokenizing input...")
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=400)
    end_time = time.time()
    print("Decoding output...")
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, end_time - start_time
