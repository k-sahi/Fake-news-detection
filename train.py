# train.py (DeBERTa-v3 version)
import sys



from transformers import AutoModelForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, Trainer, TrainingArguments

# Step 1: Load the LIAR dataset (TSV format)
df = pd.read_csv("data/train.tsv", sep='\t', header=None)
df = df[[2, 1]]  # Column 2: statement (text), Column 1: label
df.columns = ['text', 'label']

# Step 2: Clean labels (convert to binary: 0 = real, 1 = fake)
label_map = {
    "true": 0,
    "mostly-true": 0,
    "half-true": 0,
    "barely-true": 1,
    "false": 1,
    "pants-fire": 1
}
df = df[df['label'].isin(label_map)]
df['label'] = df['label'].map(label_map)

# Step 3: Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 4: Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Step 5: Load DeBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False)

def tokenize(batch):
    return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 6: Load DeBERTa model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=2)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./model/fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs"
)
print("✅ TrainingArguments created successfully!")
# Step 8: Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 9: Train the model
trainer.train()

# Step 10: Save model
trainer.save_model("./model/fine_tuned_model")
tokenizer.save_pretrained("./model/fine_tuned_model")
print("✅ Training complete. Model saved to /model/fine_tuned_model")