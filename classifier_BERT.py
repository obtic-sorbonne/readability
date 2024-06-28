import sys
import os
import random
import torch
import argparse
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from tqdm import tqdm


BERT_MODEL = 'bert-base-uncased'
LABEL_LIST = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputExample:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class Classifier:
    def __init__(self, label_list, device, cache_dir):
        self._label_list = label_list
        self._device = device
        self._tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True, cache_dir=cache_dir)
        self._model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(label_list), cache_dir=cache_dir)                                                         
        self._model.to(device)
        self._data_loader = {}  # Initialiser _data_loader comme un dictionnaire
        self._dataset = {}  # Initialiser _dataset comme un dictionnaire

    def load_data(self, set_type, examples, batch_size, max_length, shuffle):
        self._dataset[set_type] = examples
        self._data_loader[set_type] = _make_data_loader(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle)

    def get_optimizer(self, learning_rate, warmup_steps, t_total):
        self._optimizer, self._scheduler = _get_optimizer(
            self._model, learning_rate=learning_rate,
            warmup_steps=warmup_steps, t_total=t_total)

    def train_epoch(self):
        self._model.train()
        epoch_iterator = tqdm(self._data_loader['train'], desc='Training', disable=not sys.stdout.isatty())
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = self._model(**inputs)
            loss = outputs[0]
            self._optimizer.zero_grad()
            loss.backward()
            epoch_iterator.set_postfix({'loss': loss.item()})  # Remplacez accuracy par votre métrique réelle
            self._optimizer.step()
            self._scheduler.step()
           

    def evaluate(self, set_type):
        self._model.eval()
        preds_all, labels_all = [], []
        data_loader = self._data_loader[set_type]
        total_loss = 0
        for batch in data_loader:
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            with torch.no_grad():
                outputs = self._model(**inputs)
                tmp_eval_loss, logits = outputs.loss, outputs.logits
            preds = torch.argmax(logits, dim=1)
            total_loss += tmp_eval_loss.item()
            preds_all.append(preds)
            labels_all.append(inputs["labels"])
        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        accuracy = torch.sum(preds_all == labels_all).item() / labels_all.shape[0]
        return accuracy

def _get_optimizer(model, learning_rate, warmup_steps, t_total):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    return optimizer, scheduler

def _make_data_loader(examples, label_list, tokenizer, batch_size, max_length, shuffle):
    features = convert_examples_to_features(examples, tokenizer, label_list=label_list, max_length=max_length, output_mode="classification")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)  # Utilisation de 2 workers pour le chargement parallèle des données

def load_data_from_files(text_files):
    examples = {'train': [], 'dev': [], 'test': []}
    for label, file_path in text_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_lines = len(lines)
            train_split = int(0.8 * num_lines)
            dev_split = int(0.1 * num_lines) + train_split
            for i, line in enumerate(lines):
                if i < train_split:
                    set_type = 'train'
                elif i < dev_split:
                    set_type = 'dev'
                else:
                    set_type = 'test'
                examples[set_type].append(InputExample(guid=None, text_a=line.strip(), text_b=None, label=label))
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Data dir path with {train, dev, test}.txt")
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--hidden_dropout_prob', default=0.2, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_seq_length", default=20, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument('--cache', default="transformers_cache", type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--min_epochs', default=0, type=int)
    parser.add_argument("--learning_rate", default=4e-3, type=float)
    parser.add_argument('--batch_size', default=4, type=int)

    args, unknown = parser.parse_known_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    text_files = {
        'A1': '/A1.txt',
        'A2': '/A2.txt',
        'B1': '/B1.txt',
        'B2': '/B2.txt',
        'C1': '/C1.txt',
        'C2': '/C2.txt'
    }

    examples = load_data_from_files(text_files)
    t_total = len(examples['train']) * args.epochs

    classifier = Classifier(label_list=LABEL_LIST, device=DEVICE, cache_dir=args.cache)
    classifier.get_optimizer(learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, t_total=t_total)

    classifier.load_data('train', examples['train'], args.batch_size, max_length=args.max_seq_length, shuffle=True)
    classifier.load_data('dev', examples['dev'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data('test', examples['test'], args.batch_size, max_length=args.max_seq_length, shuffle=False)

    best_dev_acc, final_test_acc = -1., -1.
    for epoch in range(args.epochs):
        classifier.train_epoch()
        dev_acc = classifier.evaluate('dev')

        if epoch >= args.min_epochs:
            do_test = (dev_acc > best_dev_acc)
            best_dev_acc = max(best_dev_acc, dev_acc)
        else:
            do_test = False

        print(f'Epoch {epoch}, Dev Acc: {dev_acc * 100:.2f}, Best Ever: {best_dev_acc * 100:.2f}')

        if do_test:
            final_test_acc = classifier.evaluate('test')
            print(f'Test Acc: {final_test_acc * 100:.2f}')

    print(f'Final Dev Acc: {best_dev_acc * 100:.2f}, Final Test Acc: {final_test_acc * 100:.2f}')


'''
#!pip uninstall torch transformers accelerate

#!pip install transformers
#!pip install datasets
#!pip install torch torchvision torchaudio

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Charger le dataset JSON
with open("/cecrl_dataset.json", "r", encoding="utf-8") as json_file:
    dataset = json.load(json_file)

# Convertir le dataset en format compatible avec Hugging Face Datasets
hf_dataset = Dataset.from_dict({
    "input_text": [entry["input"] for entry in dataset],
    "label": [entry["output"] for entry in dataset]
})

# Mapper les niveaux CECRL à des labels numériques
label_to_id = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
hf_dataset = hf_dataset.map(lambda example: {"label": label_to_id[example["label"]]})

# Charger le tokenizer et le modèle pré-entraîné
model_name = "distilbert-base-uncased"  # Vous pouvez choisir un autre modèle compatible avec votre tâche
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_to_id))

# Tokenizer les données
def tokenize_function(examples):
    return tokenizer(examples["input_text"], padding="max_length", truncation=True)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.05,
)

# Créer l'objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Fine-tuner le modèle
trainer.train()

# Enregistrer le modèle fine-tuné
model.save_pretrained("./cefr_model")
tokenizer.save_pretrained("./cefr_model")

print("Fine-tuning completed and model saved!")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Charger le modèle et le tokenizer
model_path = "./cefr_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)



import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report

# Charger le dataset JSON
with open("/cecrl_dataset.json", "r", encoding="utf-8") as json_file:
    dataset = json.load(json_file)

# Convertir le dataset en format compatible avec Hugging Face Datasets
hf_dataset = Dataset.from_dict({
    "input_text": [entry["input"] for entry in dataset],
    "label": [entry["output"] for entry in dataset]
})

# Mapper les niveaux CECRL à des labels numériques
label_to_id = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
hf_dataset = hf_dataset.map(lambda example: {"label": label_to_id[example["label"]]})

# Charger le tokenizer et le modèle pré-entraîné
model_path = "./cefr_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Tokenizer les données
def tokenize_function(examples):
    return tokenizer(examples["input_text"], padding="max_length", truncation=True)

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# Définir les prédictions sur le dataset de test
def get_predictions(model, tokenized_dataset):
    input_ids = torch.tensor(tokenized_dataset['input_ids'])
    attention_mask = torch.tensor(tokenized_dataset['attention_mask'])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

# Obtenir les prédictions
predictions = get_predictions(model, tokenized_dataset)

# Convertir les prédictions en labels
predicted_labels = [list(label_to_id.keys())[pred] for pred in predictions]

# Calculer le rapport de classification
true_labels = [list(label_to_id.keys())[label] for label in tokenized_dataset['labels']]
report = classification_report(true_labels, predicted_labels, target_names=label_to_id.keys())

# Afficher le rapport de classification
print(report)


'''
