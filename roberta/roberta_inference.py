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

def load_data(data_name):
    # we only need the Qs and Q IDs so choice of file does not matter!
    # all the same Qs are posed to all the same LLM models
    if data_name == "qald9":
        questions_file = "processed_qald_9_experiment_report_qwen1.5b_none.csv"
    elif data_name == "vquanda":
        questions_file = "gemma2b.csv"
    else:
        assert False, f"unknown inference dataset: {data_name}"

    data_path = f"data/raw/{data_name}/"
    path = os.path.join(data_path, questions_file)

    df = pd.read_csv(path)
    df = df.sort_values(by=['question_id']) # sort so we have consistent order in all cases
    question_col = df["question"]
    question_id_col = df["question_id"]

    df = pd.DataFrame(
        [
            question_col,
            question_id_col,
        ]
    ).transpose().drop_duplicates()
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
            print(*row, sep='\t', file=out)

def main():
    base_data = sys.argv[1]
    data_to_infer_on = sys.argv[2]
    metric = sys.argv[3]
    out_file = sys.argv[4]
    model, tokenizer, model_path = load_model(base_data, metric)
    questions_df = load_data(data_to_infer_on)

    with open(out_file, "w") as out:
        out_data = []
        print('\t'.join(["Question", "Question_ID", "ROBERTA_Prediction", "ROBERTA_Model"]), file=out)
        for _, row in questions_df.iterrows():
            question = row["question"]
            question_id = row["question_id"]
            roberta_pred = infer(question, model, tokenizer)
            print('\t'.join(str(x) for x in [question, question_id, roberta_pred, model_path]), file=out)

if __name__ == '__main__':
    main()
