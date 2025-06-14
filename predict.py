from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from openai import OpenAI

# Load model + tokenizer
model_path = os.path.abspath("./model/deberta_fever")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def predict_deberta(claim):
    inputs = tokenizer(claim, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()
    return ("FAKE" if label == 1 else "REAL", confidence * 100)


def llm_verification(claim, model_prediction):
    prompt = f"""Analyze this claim and compare with the model prediction:

Claim: "{claim}"
Model Prediction: {model_prediction[0]} ({model_prediction[1]:.2f}% confidence)

Please verify this prediction. Respond with:
1. Your verdict (REAL or FAKE)
2. Brief explanation (1-2 sentences)
3. Confidence level (High/Medium/Low)"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def main():
    claim = input("Enter a claim to verify: ").strip()

    print("\n🤖 DeBERTa Model Analysis:")
    model_pred = predict_deberta(claim)
    print(f"Prediction: {model_pred[0]} ({model_pred[1]:.2f}% confidence)")

    print("\n🧠 LLM Verification:")
    llm_response = llm_verification(claim, model_pred)
    print(llm_response)


if __name__ == "__main__":
    main()