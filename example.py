import torch
from models.runModel import load_model, generate_sparql

# Example usage
if __name__ == "__main__":
    model_name = "google/gemma-2-2b-it"  
    quantization = "none"  # Choose quantization level: 'none', '4bit', or '8bit'

    # Load the model and tokenizer
    print(f"Loading model {model_name} with {quantization} quantization...")
    model, tokenizer = load_model(model_name, quantization=quantization)
    
    # Define the input question
    question = "What is the capital city of France?"
    input_text = (
        "Generate a SPARQL query for the input question for the DBpedia Knowledge Graph. "
        "Ensure that the query uses proper SPARQL syntax, includes prefixes, and retrieves unique results. "
        "Step-by-step: Identify relevant properties and structure the query accordingly.\n"
        "Output only the SPARQL query\n"
        f"Question: {question}\n"
    )
    
    # Generate SPARQL query
    print(f"Generating SPARQL query for question: {question}")
    generated_query, time_taken = generate_sparql(model, tokenizer, input_text)
    
    # Display the results
    print(f"\nGenerated SPARQL Query:\n{generated_query}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    
    # Free up memory
    del model, tokenizer
    torch.cuda.empty_cache()
