#!pip install -qqq -U transformers datasets accelerate peft trl bitsandbytes wandb --progress-bar off

import gc
import os
import torch
import wandb
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_data_from_files(text_files):
    examples = {'train': [], 'test': []}
    for label, file_path in text_files.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_lines = len(lines)
            train_split = int(0.8 * num_lines)
            for i, line in enumerate(lines):
                set_type = 'train' if i < train_split else 'test'
                examples[set_type].append({'text': line.strip(), 'label': label})
    return examples

text_files = {
    'A1': '/A1.txt',
    'A2': '/A2.txt',
    'B1': '/B1.txt',
    'B2': '/B2.txt',
    'C1': '/C1.txt',
    'C2': '/C2.txt'
}

examples = load_data_from_files(text_files)

print(f"Nombre d'exemples d'entraînement : {len(examples['train'])}")
print(f"Nombre d'exemples de test : {len(examples['test'])}")

dataset = DatasetDict({
    'train': Dataset.from_list(examples['train']),
    'test': Dataset.from_list(examples['test'])
})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

base_model = "alpindale/Mistral-7B-v0.2-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)
model = PeftModel(model, peft_config)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    run_name="mistral_cefr_classification"
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate(eval_dataset=dataset['test'])
print(eval_results)
