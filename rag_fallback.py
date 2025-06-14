import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from serpapi import GoogleSearch
from openai import OpenAI

# Load model + tokenizer
model_path = os.path.abspath("./model/deberta_fever")
checkpoint_path = os.path.join(model_path, "checkpoint-3072")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, local_files_only=True)
model.eval()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def predict_deberta(claim):
    inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()
    return ("FAKE" if label == 1 else "REAL", confidence * 100)

def search_google(query):
    params = {
        "q": query,
        "api_key": st.secrets.get("SERPAPI_API_KEY"),
        "engine": "google",
        "num": 5
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = [r["snippet"] for r in results.get("organic_results", []) if "snippet" in r]
    return "\n".join(snippets)

def llm_verification(claim, context):
    prompt = f"""Claim: "{claim}"
Based on the following web evidence, is this claim true or fake?

Web Evidence:
{context}

Answer only TRUE or FAKE, and explain briefly."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def main():
    claim = input("Enter a claim to verify: ").strip()

    label, conf = predict_deberta(claim)
    print(f"🤖 DeBERTa model says: {label} ({conf:.2f}%)")

    if conf < 95 and label == "FAKE":
        print("🔁 Triggering LLM + Web Search fallback...")
        context = search_google(claim)
        verdict = llm_verification(claim, context)
        print("\n🧠 LLM verdict:\n" + verdict)

if __name__ == "__main__":
    main()
