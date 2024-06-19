#!pip install transformers torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from huggingface_hub import notebook_login

#notebook_login()
#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


# Charger le modèle et le tokenizer T5
model_name = "dbddv01/gpt2-french-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def reformulate_text(text, level, model, tokenizer, max_length=200):
    # Prompt plus détaillé pour aider le modèle à comprendre la tâche
    prompt = f" Reformulez ce texte au niveau {level} : {text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True)

    # Génération du texte
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    reformulated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reformulated_text

# Exemple d'utilisation
text = "On ne peut plus repenser le contenu d’une culture scolaire commune en se référant à ce qui existe ou a existé. Nous vivons dans un univers bouleversé par les révolutions scientifiques et technologiques."
level = "A1"  # Niveau CECRL cible

reformulated_text = reformulate_text(text, level, model, tokenizer)
print(f"Texte original : {text}")
print(f"Texte reformulé au niveau {level} : {reformulated_text}")
