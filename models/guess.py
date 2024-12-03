from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import json

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
    
def load_data_split(data, split_name, num_labels):
    queries = []
    labels = []
    for item_id in data[split_name]:
        queries.append(data["train"][item_id]["query"])
        label_idx = data["train"][item_id]["class"]
        labels_i = [0 for _ in range(num_labels)]
        labels_i[label_idx] = 1
        labels.append(labels_i)
    return queries, labels

def load_data(tokenizer):
    with open('../data/roberta-data-tmp.json', 'r') as file:
        data = json.load(file)

    # detect number of labels
    classes = set()
    for item_id in data['train']:
        classes.add(data["train"][item_id]["class"])
    num_labels = len(classes)

    # load queries and labels
    train_queries, train_labels = load_data_split(data, "train", num_labels)
    test_queries, test_labels = load_data_split(data, "test", num_labels)
    valid_queries, valid_labels = load_data_split(data, "valid", num_labels)

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

# run finetuning (PyTorch)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# model.train()
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
# for epoch in range(3):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()
# model.eval()

# run finetuning (Tranformers)
training_args = TrainingArguments(
    output_dir='./roberta/results',          # output directory
    num_train_epochs=3,                      # total number of training epochs
    per_device_train_batch_size=16,          # batch size per device during training
    per_device_eval_batch_size=64,           # batch size for evaluation
    warmup_steps=500,                        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                       # strength of weight decay
    logging_dir='./roberta/logs',            # directory for storing logs
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()
