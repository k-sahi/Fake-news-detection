# 📰 Fake News Detection (Base Version)

This project is an **Advanced Fake News Detection System** built on top of fine-tuned **DeBERTa-v3** models using the **FEVER dataset**, enhanced with:

- **LLM Verification** via OpenAI's GPT models
- **Fact Database** cross-checks
- **RAG Fallback** with live Google Search (via SerpAPI)

## ✅ Features

- **Model**: FEVER-trained `DeBERTa-v3-small`
- **Frontend**: Streamlit web interface
- **Backend**: FastAPI endpoint (`/predict`)
- **Verification**: OpenAI GPT-4-turbo for enhanced fact verification
- **Fallback**: RAG-based real-time search when confidence is low

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Fake_News_Detection_FDM.git
cd Fake_News_Detection_FDM
