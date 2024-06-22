#!pip install sacremoses

import pandas as pd
import numpy as np
from transformers import FlaubertTokenizer, FlaubertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import re


def text_into_tab_de_phrase(text):
    sentence_pattern = re.compile(
        r'(?:[A-Z][^.!?]*[.!?]|[A-Z][^.!?]*\.\.\.)'
    )
    sentences = sentence_pattern.findall(text)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences



files = {
    "A1": '/A1.txt',
    "A2": '/A2.txt',
    "B1": '/B1.txt',
    "B2": '/B2.txt',
    "C1": '/C1.txt',
    "C2": '/C2.txt'
}



data = {"text": [], "label": []}
label_mapping = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

for level, filename in files.items():
    with open(filename, 'r') as f:
        phrases = text_into_tab_de_phrase(f.read())
        data["text"].extend(phrases)
        data["label"].extend([label_mapping[level]] * len(phrases))
        print(phrases)

df = pd.DataFrame(data)

df = df.sample(frac=0.1, random_state=42) #frac est la prportion utilise du dataset


train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=64, truncation=True, padding='max_length')  # Shorter max_length

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


model = FlaubertForSequenceClassification.from_pretrained('flaubert/flaubert_base_cased', num_labels=6)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=12,  
    per_device_eval_batch_size=12,   
    num_train_epochs=1,             
    weight_decay=0.05,
    save_total_limit=1,             
    save_steps=1000,               
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

trainer.train()


predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)


print(f'Accuracy: {accuracy_score(labels, predictions):.2f}')
print(classification_report(labels, predictions, target_names=list(label_mapping.keys())))

