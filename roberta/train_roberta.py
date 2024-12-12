from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import os
import sys
import glob

'''
refs:
    https://huggingface.co/FacebookAI/roberta-large
    https://huggingface.co/docs/transformers/main/en/model_doc/roberta#transformers.RobertaForSequenceClassification
    https://huggingface.co/transformers/v3.2.0/custom_datasets.html
'''

class GuesserDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

def process_labels(label_idxs, num_labels):
    labels = []
    for label_idx in label_idxs:
        labels_i = [0 for _ in range(num_labels)]
        labels_i[label_idx] = 1
        labels.append(labels_i)
    return labels

def load_data(path, tokenizer):
    train_file_path = glob.glob(os.path.join(path, "*.train.csv"))
    assert len(train_file_path) == 1, f"exactly 1 file should be found, found {len(train_file_path)}"
    train_file_path = train_file_path[0]
    test_file_path = glob.glob(os.path.join(path, "*.test.csv"))
    assert len(test_file_path) == 1, f"exactly 1 file should be found, found {len(train_file_path)}"
    test_file_path = test_file_path[0]

    data_train = pd.read_csv(train_file_path).set_axis(["question", "level"], axis=1)
    train_queries = data_train["question"].to_list()
    num_labels = len(set(data_train["level"]))
    train_labels = process_labels(data_train["level"].to_list(), num_labels)

    data_test = pd.read_csv(test_file_path).set_axis(["question", "level"], axis=1)
    test_queries = data_test["question"].to_list()
    # test_labels = process_labels(data_test["level"].to_list(), num_labels)
    
    # data_valid = pd.read_csv("data/processed/qald9.valid").set_axis(["question", "level"], axis=1)
    # valid_queries = data_valid["question"].to_list()
    # valid_labels = process_labels(data_valid["level"].to_list(), num_labels)

    # encode queries
    train_encodings = tokenizer(train_queries, truncation=True, padding=True)
    # test_encodings = tokenizer(test_queries, truncation=True, padding=True)
    # val_encodings = tokenizer(valid_queries, truncation=True, padding=True)

    # create dataset objects
    train_dataset = GuesserDataset(train_encodings, train_labels)
    # test_dataset = GuesserDataset(test_encodings, test_labels)
    # valid_dataset = GuesserDataset(val_encodings, valid_labels)
    test_dataset = (test_queries, data_test["level"])
    # valid_dataset = (valid_queries, valid_labels)

    return train_dataset, test_dataset, num_labels

def train(train_dataset, num_labels, save_path):
    # load model
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-large',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    # run finetuning (Tranformers)
    training_args = TrainingArguments(
        output_dir=os.path.join(save_path, 'results'),
        num_train_epochs=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(save_path, 'logs'),
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    return model

def infer(model, tokenizer, nl_query):
    tokenised_query = tokenizer(nl_query, return_tensors='pt').to('cuda')
    guess = model(**tokenised_query).logits
    level_guess = int(torch.argmax(guess))
    return level_guess
    
def eval(model, tokenizer, test_dataset, save_path):
    # load raw data
    test_queries, test_labels = test_dataset
    labels_true = []
    labels_pred = []
    for i in range(len(test_queries)):
        query = test_queries[i]
        label_true = int(test_labels[i])
        labels_true.append(label_true)
        label_pred = int(infer(model, tokenizer, query))
        labels_pred.append(label_pred)

    # output labels and convert to torch
    labels_true = torch.tensor(labels_true)
    labels_pred = torch.tensor(labels_pred)
    print("True:", labels_true.tolist())
    print("Pred:", labels_pred.tolist())
    print("Equa:", [int(x) for x in (labels_true == labels_pred)])

    # evaluation metrics
    accuracy = torch.sum(labels_true == labels_pred) / len(test_queries)
    print("Accuracy: ", round(float(accuracy)), 4)

def main(data_path, save_path):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    train_dataset, test_dataset, num_labels = load_data(data_path, tokenizer)
    model = train(train_dataset, num_labels, save_path)
    eval(model, tokenizer, test_dataset, save_path)

if __name__ == "__main__":
    data_path = sys.argv[1]
    save_path = sys.argv[2]
    main(data_path, save_path)
