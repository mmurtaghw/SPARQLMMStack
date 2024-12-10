import json
from models.runModel import load_model, generate_sparql, validate_sparql
from roberta.guesser import Guesser

class MinGuesser(Guesser):
    def guess(self, nl_query):
        """
        Always returns the minimum level.

        Args:
            nl_query (str): The query to be posed to the LLM chain.

        Returns:
            int: The minimum level.
        """
        return self.min_level


def load_qald_question(qald_data, lang="en"):
    """
    Extract the first question in the specified language from QALD-9 data.

    Args:
        qald_data (dict): The QALD-9 dataset loaded as a dictionary.
        lang (str): Language of the question to extract.

    Returns:
        tuple: The question and its SPARQL query.
    """
    for question_data in qald_data["questions"]:
        for q in question_data["question"]:
            if q["language"] == lang:
                return q["string"], question_data["query"]["sparql"]
    return None, None


def main():
    # Load QALD-9 data
    with open("data/QALD_9_plus/qald_9_plus_test_dbpedia.json", "r") as file:
        qald_data = json.load(file)

    # Load the LLM stack
    llm_stack = [
        {"model_name": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": "none"},
        {"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "quantization": "none"}
    ]

    # Get a question and its reference SPARQL query
    question, reference_sparql = load_qald_question(qald_data)

    # Initialize MinGuesser
    min_guesser = MinGuesser(min_level=1, max_level=len(llm_stack))

    # Determine starting LLM level
    current_level = min_guesser.guess(question)

    # Generate and validate a SPARQL query
    for level in range(current_level, len(llm_stack) + 1):
        llm = llm_stack[level - 1]
        model, tokenizer = load_model(llm["model_name"], llm["quantization"])

        # Generate SPARQL query
        input_text = (
            f"Generate a SPARQL query for the question: {question} "
            "using proper syntax and targeting DBpedia."
        )
        generated_query, time_taken = generate_sparql(model, tokenizer, input_text)
        print(f"Level {level}: Generated Query:\n{generated_query}\n")

        # Validate SPARQL query using the next LLM
        is_correct, validation_time = validate_sparql(model, tokenizer, question, generated_query)

        if is_correct:
            print(f"Level {level}: Query validated as correct.")
            with open("results.json", "w") as results_file:
                json.dump(
                    {
                        "level": level,
                        "question": question,
                        "generated_query": generated_query,
                        "generation_time": time_taken,
                        "validation_time": validation_time,
                    },
                    results_file,
                    indent=4,
                )
            return
        else:
            print(f"Level {level}: Query deemed incorrect, escalating to next LLM.")

    print("Query validation failed at all levels.")


if __name__ == "__main__":
    main()
