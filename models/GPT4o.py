import os
import re
import openai
from dotenv import load_dotenv

# Load environment variables from a file named 'env'
load_dotenv(dotenv_path="env")
openai.api_key = os.getenv("GPT4_API_KEY")


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

def generate_oracle_sparql(prompt):
    """
    Use GPT-4o API to generate a SPARQL query based on the provided prompt.

    Args:
        prompt (str): The prompt to guide the SPARQL query generation.

    Returns:
        str: The generated SPARQL query.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a SPARQL query generation assistant. You only generate valid SPARQL queries. Do not provide explanations or extra text."},
            {"role": "user", "content": f"{prompt}\n\nReturn only the SPARQL query enclosed within triple backticks (```sparql ... ```)."}
        ]

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        # Access the content of the first choice
        raw_output = response.choices[0].message.content
        print(raw_output)

        # Extract and return the SPARQL query
        return extract_sparql(raw_output)
    except Exception as e:
        print(f"Oracle model failed to generate SPARQL query: {e}")
        return None



if __name__ == "__main__":
    # Test GPT-4o API
    test_question = "What is the time zone of Salt Lake City?"
    test_prompt = (
        f"Generate a SPARQL query for the following question, ensuring proper syntax:\n\n"
        f"using proper syntax and targeting DBpedia."
        f"Question: {test_question}\n\n"
        "SPARQL Query:"
    )
    
    print("Testing GPT-4o API...")
    sparql_query = generate_oracle_sparql(test_prompt)
    
    if sparql_query:
        print(f"Generated SPARQL Query:\n{sparql_query}")
    else:
        print("Failed to generate SPARQL query.")
