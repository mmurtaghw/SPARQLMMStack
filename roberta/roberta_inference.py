from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import sys
import pandas as pd
import os

def load_qald_9_data():
    llm_stack = { # dictionaries preserve order!
        ("qwen0.5b"): "processed_qald_9_experiment_report_qwen0.5b_none (1).csv",
        ("qwen1.5b"): "processed_qald_9_experiment_report_qwen1.5b_none.csv",
        ("Gemma2b"): "processed_qald_9_experiment_report_Gemma2b_none.csv",
        ("mistral7b"): "processed_qald_9_experiment_report_mistral7b_none.csv",
        ("Gemma9b"): "processed_qald_9_experiment_report_Gemma9b_none (1).csv"
    }
    return llm_stack

def load_quanda_data():
    llm_stack = { # dictionaries preserve order!
        ("qwen0.5b"): "Qwen0.5B.csv",
        ("qwen1.5b"): "Qwen1.5B.csv",
        ("Gemma2b"): "gemma2b.csv",
        ("mistral7b"): "Mistral7B.csv",
        ("Gemma9b"): "gemma9b.csv"
    }
    return llm_stack

def load_model(data, metric):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    if data == "qald9":
        model_path = f'./output/{data}/{metric}/results/checkpoint-400/'
    elif data == "vquanda":
        model_path = f'./output/{data}/{metric}/results/checkpoint-2500/'
    else:
        assert False, f"invalid data tag: {data}"
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer, model_path

def load_data(llm_name, data_name):
    if data_name == "qald9":
        llm_stack= load_qald_9_data()
    elif data_name == "vquanda":
        llm_stack = load_quanda_data()
    else:
        assert False, f"unknown inference dataset: {data_name}"
        
    data_path = f"data/raw/{data_name}/"
    path = os.path.join(data_path, llm_stack[llm_name])
    df = pd.read_csv(path)
    df = df.sort_values(by=['question_id']) # sort so we have consistent order in all cases
    question_col = df["question"]
    question_id_col = df["question_id"]

    df = pd.DataFrame(
        [
            question_col,
            question_id_col,
        ]
    ).transpose()
    df = df.set_axis(
        [
            "question",
            "question_id"
        ], axis=1
    )
    return df

def infer(nl_query, model, tokenizer):
    tokenised_query = tokenizer(nl_query, return_tensors='pt')
    guess = model(**tokenised_query).logits
    level_guess = int(torch.argmax(guess))
    return level_guess

def write_data(out_file, out_data):
    with open(out_file, "w") as out:
        for row in out_data:
            print(*row, sep=',', file=out)

def main():
    base_data = sys.argv[1]
    data_to_infer_on = sys.argv[2]
    metric = sys.argv[3]
    llm_name = sys.argv[4]
    out_file = sys.argv[5]
    model, tokenizer, model_path = load_model(base_data, metric)
    questions_df = load_data(llm_name, data_to_infer_on)

    out_data = []
    out_data.append(["Question ID", "Question", "ROBERTA_Prediction", "ROBERTA_Model"])
    for _, row in questions_df.iterrows():
        question = row["question_id"]
        question_id = row["question"]
        roberta_pred = infer(question, model, tokenizer)
        out_data.append([question, question_id, roberta_pred, model_path])
    write_data(out_file, out_data)