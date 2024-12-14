import json
import re
import torch
from models.runModel import load_model, generate_sparql, validate_sparql
from roberta.guesser import MinGuesser


def load_qald_questions(qald_data, lang="en", start=0, end=None):
    """
    Extract questions and their IDs in the specified language from QALD-9 data within a given range.

    Args:
        qald_data (dict): The QALD-9 dataset loaded as a dictionary.
        lang (str): Language of the questions to extract.
        start (int): Starting index of questions to process.
        end (int): Ending index of questions to process.

    Returns:
        list: A list of dictionaries containing question ID, question string, and reference SPARQL query.
    """
    questions = []
    for question_data in qald_data["questions"]:
        question_id = question_data["id"]
        for q in question_data["question"]:
            if q["language"] == lang:
                questions.append({
                    "id": question_id,
                    "string": q["string"],
                    "reference_sparql": question_data["query"]["sparql"]
                })
                break  # Ensure only one question per entry (language match)
    return questions[start:end]


def extract_sparql(query_text):
    """
    Extract SPARQL queries from the text. Handles multiple queries in the same block.

    Args:
        query_text (str): Text containing potential SPARQL queries.

    Returns:
        str: The extracted SPARQL query or None if no query is found.
    """
    if not query_text or not isinstance(query_text, str):
        return None

    # Match SPARQL blocks enclosed in ```sparql
    queries = re.findall(r"```sparql(.*?)```", query_text, re.DOTALL)

    # If no ```sparql delimiter is found, look for common SELECT/ASK/CONSTRUCT patterns
    if not queries:
        queries = re.findall(
            r"(?:(?:PREFIX|SELECT|ASK|CONSTRUCT).*?WHERE\s*{.*?})",
            query_text,
            re.DOTALL | re.IGNORECASE,
        )

    # Clean up extracted queries (remove leading/trailing whitespace)
    queries = [query.strip() for query in queries]

    # Return queries as a concatenated string if multiple exist
    return "\n\n".join(queries) if queries else None


def main(start=0, end=None):
    # Load QALD-9 data
    with open("data/QALD_9_plus/data/qald_9_plus_test_dbpedia.json", "r") as file:
        qald_data = json.load(file)

    # Load the LLM stack
    llm_stack = [
        {"model_name": "google/gemma-2-9b-it", "quantization": "none"},
        {"model_name": "mistralai/Mistral-7B-Instruct-v0.3", "quantization": "none"},
        {"model_name": "google/gemma-2-2b-it", "quantization": "none"},
        {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": "none"},
        {"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "quantization": "none"}
    ]

    # Extract questions from the dataset
    questions = load_qald_questions(qald_data, start=start, end=end)

    # Initialize MinGuesser
    min_guesser = MinGuesser(min_level=1, max_level=len(llm_stack))

    # Initialize results
    results = {
        "success": [],
        "failure": [],
        "details": []
    }

    # Process each question
    for idx, question_data in enumerate(questions, start=start):
        question_id = question_data["id"]
        question = question_data["string"]
        reference_sparql = question_data["reference_sparql"]

        print(f"Processing Question ID {question_id}: {question}")
        current_level = min_guesser.guess(question)

        # Track generated queries for this question
        generated_queries = []

        # Generate and validate a SPARQL query
        for level in range(current_level, len(llm_stack) + 1):
            llm = llm_stack[level - 1]
            model, tokenizer = load_model(llm["model_name"], llm["quantization"])

            try:
                # Generate SPARQL query
                input_text = (
                    f"Generate a SPARQL query for the question: {question} "
                    "using proper syntax and targeting DBpedia."
                )
                raw_generated_query, time_taken = generate_sparql(model, tokenizer, input_text)

                # Extract SPARQL query
                generated_query = extract_sparql(raw_generated_query)
                generated_queries.append({
                    "level": level,
                    "raw_query": raw_generated_query,
                    "extracted_query": generated_query,
                    "generation_time": time_taken
                })

                print(f"Level {level}: Extracted SPARQL Query:\n{generated_query}\n")

                if not generated_query:
                    print(f"Level {level}: Failed to extract SPARQL query. Escalating to next LLM.")
                    continue  # Escalate to the next LLM if extraction fails

                # Validate SPARQL query using the next LLM
                is_correct, validation_time = validate_sparql(model, tokenizer, question, generated_query)

                if is_correct:
                    print(f"Level {level}: Query validated as correct.")
                    results["success"].append({
                        "question_id": question_id,
                        "question": question,
                        "reference_query": reference_sparql,
                        "validated_query": generated_query,
                        "level": level,
                        "generation_time": time_taken,
                        "validation_time": validation_time
                    })
                    results["details"].append({
                        "question_id": question_id,
                        "question": question,
                        "queries": generated_queries
                    })
                    break
                else:
                    print(f"Level {level}: Query deemed incorrect, escalating to next LLM.")
            finally:
                # Clear model and tokenizer from GPU memory
                del model, tokenizer
                torch.cuda.empty_cache()

        else:
            # Record failure if no valid query is found
            results["failure"].append({
                "question_id": question_id,
                "question": question,
                "reference_query": reference_sparql,
                "queries": generated_queries
            })
            results["details"].append({
                "question_id": question_id,
                "question": question,
                "queries": generated_queries
            })
            print(f"Question ID {question_id}: Validation failed at all levels.")

    # Write results to JSON file
    with open("results.json", "w") as results_file:
        json.dump(results, results_file, indent=4)


if __name__ == "__main__":
    # Configure start and end indices for processing
    start_idx = 0
    end_idx = 5  # Set a specific number or keep None to process all
    main(start=start_idx, end=end_idx)
