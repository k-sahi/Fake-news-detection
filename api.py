# api.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = FastAPI()

# Load model
model_path = "./model/deberta_fever"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model.eval()

# Request format
class NewsRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_fake_news(request: NewsRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

    label_map = {0: "REAL", 1: "FAKE"}
    return {"prediction": label_map[predicted_class]}