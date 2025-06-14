from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load and preprocess data
dataset = load_dataset("fever", "v1.0", split="train[:22900]")
val_dataset = load_dataset("fever", "v1.0", split="labelled_dev[:5725]")

def format(example):
    return {
        "text": f"{example['claim']} [SEP] {example['label']}",
        "label": 0 if example["label"] == "REFUTES" else 1
    }

dataset = dataset.map(format)
val_dataset = val_dataset.map(format)

tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-small")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define model
model = DebertaV2ForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)

# ✅ Only use arguments supported in v4.52.4
training_args = TrainingArguments(
    output_dir="./fever_deberta_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=1,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset
)

trainer.train()
# ✅ Save final trained model and tokenizer
model_path = "./model/deberta_fever"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

print("✅ Final model saved to", model_path)