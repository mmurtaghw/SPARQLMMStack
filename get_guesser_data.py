import json
from models.runModel import load_model, generate_sparql
import torch
from nltk.translate.bleu_score import sentence_bleu
from rdflib import ConjunctiveGraph
from SPARQLWrapper import SPARQLWrapper, JSON

def load_qald_9_plus():
    try:
        with open('data/QALD_9_plus/data/qald_9_plus_test_dbpedia.json', 'r') as file:
            qald_data = json.load(file)
        return qald_data
    except:
        raise

def calculate_bleu_score(reference_query, generated_query):
    """Calculate BLEU score comparing reference and generated queries."""
    reference = reference_query.split()
    candidate = generated_query.split()
    return sentence_bleu([reference], candidate)

def validate_sparql_syntax(query):
    """Validate the syntax of a SPARQL query."""
    try:
        g = ConjunctiveGraph()
        g.query(query)
        return True
    except Exception:
        return False
    
def check_query_execution(query, endpoint="https://dbpedia.org/sparql"):
    """Check if the query executes successfully on the endpoint."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        sparql.query().convert()
        return True
    except Exception:
        return False

def forumate_prompt(question):
    '''
    Formualtes a prompt given a natural-langauge question

    Args:
        question (str): the question that should be formualted into an LLM prompt in a structured format.

    Returns:
        str: the formatted prompt to the LLM
    '''
    input_text = (
        "Generate a SPARQL query for the input question for the DBpedia Knowledge Graph. "
        "Ensure that the query uses proper SPARQL syntax, includes prefixes, and retrieves unique results. "
        "Step-by-step: Identify relevant properties and structure the query accordingly.\n"
        "Output only the SPARQL query\n"
        f"Question: {question}\n"
    )
    return input_text

def ask_oracle(oracle_llm, question, sparql_pred):
    model_name = oracle_llm["model_name"]
    quantization = oracle_llm["quantization"]
    model, tokenizer = load_model(model_name, quantization=quantization)
    prompt = {
        f'The following SPARQL quuery was formualted for the DBpedia Knowledge Graph to answer the question "{question}".'
        f'Query: "{sparql_pred}"'
        'Is this the correct formulation of a SPARQL query to answer the given question?'
        'Please respond only with "Yes" or "No".'
    }
    response, time_taken = generate_sparql(model, tokenizer, prompt)
    return response

def write_results(results_dict, out_name):
    '''
    Write all results to an output json file.

    Args:
        results_dict (dict): the dictionary of nresults that should be written out.
        out_name (str): the path to save results.

    Returns:
        None
    '''
    try:
        with open(out_name, 'w') as f:
            json.dump(results_dict, f)
    except:
        raise

def get_query_results(query, endpoint="https://dbpedia.org/sparql"):
    """Get the response to the queyr on the endpoint."""
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        '''
        format
        
        For "SELECT * WHERE {?s ?p ?o. ?p ?s ?o.} LIMIT 1") 
        {
            'head': {
                'link': [],
                'vars': ['s', 'p', 'o']
            },
            'results': {
                'distinct': False,
                'ordered': True,
                'bindings': []
            }
        }

        For "PREFIX res: <http://dbpedia.org/resource/> PREFIX dbp: <http://dbpedia.org/property/> SELECT DISTINCT ?uri WHERE { res:Salt_Lake_City <http://dbpedia.org/ontology/timeZone> ?uri }"
        {
            'head': {
                'link': [],
                'vars': ['uri']
            }, 'results': {
                'distinct': False,
                'ordered': True,
                'bindings': [
                    {
                        'uri': {
                            'type': 'uri',
                            'value': 'http://dbpedia.org/resource/Mountain_Time_Zone'
                        }
                    }
                ]
            }
        }
        '''
        x = sparql.query().convert()
        query_runs = True
        has_answers = len(x["results"]["bindings"]) > 0
    except Exception:
        query_runs = False
        has_answers = False
    return query_runs, has_answers

def main():
    qald_data = load_qald_9_plus()
    lang = "en"
    llm_stack = [
        {
            "model_name": "google/gemma-2-2b-it",
            "quantization": "none" # Choose quantization level: 'none', '4bit', or '8bit'
        },
    ]
    out_name = "out.json"

    results = {}
    for llm in llm_stack:
        model_name = llm["model_name"]
        quantization = llm["quantization"]
        model, tokenizer = load_model(model_name, quantization=quantization)
        assert not (model_name, quantization) in results, f"Key conflict with model name and quantisatiton: {model_name, quantization}"
        results[(model_name, quantization)] = {}

        for data in qald_data["questions"]:
            question_id = data["id"]
            question = None
            for q in data["question"]:
                if q["language"] == lang:
                    question = q["string"]
            assert question is not None, f"could not find an Q in English, why? ID: {question_id}"

            sparql_true = data["query"]
            input_text = forumate_prompt(question)
            sparql_pred, time_taken = generate_sparql(model, tokenizer, input_text)
            
            bleu_score = calculate_bleu_score(sparql_true, sparql_pred)
            sparql_valid_syntax = validate_sparql_syntax(sparql_pred)
            sparql_runs_on_endpoint, sparql_returns_results = get_query_results(sparql_pred)

            assert not question_id in results[(model_name, quantization)], f"Key conflict with question ID: {model_name, question_id}"
            results[(model_name, quantization)][question_id] = {
                "question": question,
                "input_text": input_text,
                "sparql_true": sparql_true,
                "sparql_pred": sparql_pred,
                "time_taken": time_taken,
                "bleu_score": bleu_score,
                "sparql_valid_syntax": sparql_valid_syntax,
                "sparql_runs_on_endpoint": sparql_runs_on_endpoint,
                "sparql_returns_results": sparql_returns_results
            }

        write_results(results, out_name)
        del model, tokenizer
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
