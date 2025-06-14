from datasets import load_dataset
import json

# Use 'labelled_dev' split instead of 'train'
dataset = load_dataset("fever", "v1.0", split="labelled_dev")

def combine_text(example):
    # Flatten all evidence text into one string
    evidence = ""
    if example.get("evidence"):
        try:
            evidence = " ".join([" ".join(e) for group in example["evidence"] for e in group if isinstance(e, list)])
        except Exception:
            pass
    claim = example["claim"]
    label = example["label"]
    return {"text": f"{claim} [SEP] {evidence}", "label": label}

dataset = dataset.map(combine_text)
dataset = dataset.filter(lambda x: x["label"] in ["SUPPORTS", "REFUTES"])

# Convert textual labels to binary
label_map = {"SUPPORTS": 1, "REFUTES": 0}
formatted_data = [{"text": x["text"], "label": label_map[x["label"]]} for x in dataset]

# Save as JSON
with open("fever_finetune_data.json", "w") as f:
    json.dump(formatted_data, f, indent=2)

print("✅ fever_finetxune_data.json created successfully.")