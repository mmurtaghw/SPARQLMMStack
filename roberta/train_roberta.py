from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd

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

def load_data(tokenizer):
    data_train = pd.read_csv("data/processed/qald9.train").set_axis(["question", "level"], axis=1)
    train_queries = data_train["question"].to_list()
    num_labels = len(set(data_train["level"]))
    train_labels = process_labels(data_train["level"].to_list(), num_labels)

    data_test = pd.read_csv("data/processed/qald9.test").set_axis(["question", "level"], axis=1)
    test_queries = data_test["question"].to_list()
    test_labels = process_labels(data_test["level"].to_list(), num_labels)
    
    data_valid = pd.read_csv("data/processed/qald9.valid").set_axis(["question", "level"], axis=1)
    valid_queries = data_valid["question"].to_list()
    valid_labels = process_labels(data_valid["level"].to_list(), num_labels)

    # encode queries
    train_encodings = tokenizer(train_queries, truncation=True, padding=True)
    test_encodings = tokenizer(test_queries, truncation=True, padding=True)
    val_encodings = tokenizer(valid_queries, truncation=True, padding=True)

    # create dataset objects
    train_dataset = GuesserDataset(train_encodings, train_labels)
    test_dataset = GuesserDataset(test_encodings, test_labels)
    valid_dataset = GuesserDataset(val_encodings, valid_labels)

    return train_dataset, test_dataset, valid_dataset, num_labels

# load data
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
train_dataset, test_dataset, valid_dataset, num_labels = load_data(tokenizer)

# load model
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-large',
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# run finetuning (Tranformers)
training_args = TrainingArguments(
    output_dir='./output/results',          # output directory
    num_train_epochs=50,                      # total number of training epochs
    per_device_train_batch_size=16,          # batch size per device during training
    per_device_eval_batch_size=64,           # batch size for evaluation
    warmup_steps=500,                        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                       # strength of weight decay
    logging_dir='./output/logs',            # directory for storing logs
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()
