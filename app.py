import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import os
import re
from datetime import datetime

# Initialize with strict configuration
st.set_page_config(
    page_title="Fake News Detector Pro+",
    page_icon="🔍",
    layout="wide"
)

# Enhanced CSS for better readability
st.markdown("""
<style>
    /* Improved metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .metric-card.warning {
        border-left-color: #FFC107;
    }
    .metric-card.error {
        border-left-color: #F44336;
    }

    /* Typography improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Better input styling */
    .stTextArea textarea {
        min-height: 120px !important;
        font-size: 1rem !important;
    }

    /* Enhanced discrepancy alerts */
    .discrepancy-alert {
        background: #fff3cd;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration with accuracy focus
MODEL_NAME = "./model/deberta_fever"  # Upgraded to larger model
LLM_MODEL = "gpt-4-turbo"  # More accurate version
TEMPERATURE = 0.1  # Lower for factual consistency
MAX_CLAIM_LENGTH = 500  # Prevent overly long inputs

# Fact verification database (sample - expand with your own)
FACT_DATABASE = {
    "apollo 11": {"correct_date": "1969", "correct_verb": "landed"},
    "earth circumference": {"value": "40,075 km"},
    "boiling point of water": {"value": "100°C at sea level"}
}


# System Setup with caching
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None


tokenizer, model, device = load_model()
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))


def preprocess_claim(claim):
    """Clean and validate input claim"""
    claim = claim.strip()
    if len(claim) > MAX_CLAIM_LENGTH:
        raise ValueError(f"Claim exceeds maximum length of {MAX_CLAIM_LENGTH} characters")
    if not claim or claim.lower() == "none":
        raise ValueError("Empty claim provided")
    return claim


def check_against_database(claim):
    """Cross-reference with known facts"""
    claim_lower = claim.lower()
    results = {}

    for fact_key, fact_data in FACT_DATABASE.items():
        if fact_key in claim_lower:
            results[fact_key] = fact_data

    return results if results else None


def predict_deberta(claim):
    """Enhanced prediction with input validation"""
    try:
        inputs = tokenizer(
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            label = torch.argmax(probs).item()
            confidence = probs[0][label].item()

        return {
            "verdict": "FAKE" if label == 1 else "REAL",
            "confidence": confidence * 100,
            "error": None
        }
    except Exception as e:
        return {
            "verdict": None,
            "confidence": None,
            "error": f"Model prediction failed: {str(e)}"
        }


def llm_verification(claim, model_prediction, fact_check_results=None):
    """Enhanced verification with fact-checking"""
    try:
        fact_check_context = ""
        if fact_check_results:
            fact_check_context = "\nKNOWN FACTS:\n" + "\n".join(
                f"- {key}: {value}"
                for key, value in fact_check_results.items()
            )

        prompt = f"""Perform STRICT fact verification on this claim:

CLAIM: "{claim}"
MODEL PREDICTION: {model_prediction['verdict']} ({model_prediction['confidence']:.1f}% confidence){fact_check_context}

INSTRUCTIONS:
1. Identify ALL factual errors (dates, numbers, names, wording)
2. Check against authoritative sources
3. Provide corrected version if needed
4. List specific evidence sources
5. Rate confidence based on available evidence

RESPONSE FORMAT (EXACTLY):
Verdict: REAL/FAKE
Errors: [comma-separated list or 'None']
Correction: [accurate version or 'None']
Confidence: High/Medium/Low
Sources: [1-3 authoritative sources]
Rationale: [1 sentence explanation]"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional fact-checker. Be meticulous and cite sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=500
        )
        return parse_llm_response(response.choices[0].message.content.strip())
    except Exception as e:
        return {"error": f"LLM verification failed: {str(e)}"}


def parse_llm_response(response):
    """Strict parsing of LLM response"""
    result = {"raw": response}
    patterns = {
        "verdict": r"Verdict:\s*(REAL|FAKE)",
        "errors": r"Errors:\s*(.+?)\n",
        "correction": r"Correction:\s*(.+?)\n",
        "confidence": r"Confidence:\s*(High|Medium|Low)",
        "sources": r"Sources:\s*(.+?)\n",
        "rationale": r"Rationale:\s*(.+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result[key] = match.group(1).strip()

    return result


def display_result(claim, model_pred, llm_response, fact_check):
    """Enhanced result display with more details"""
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.subheader("🤖 AI Model Analysis")
            if model_pred["error"]:
                st.error(model_pred["error"])
            else:
                st.metric(
                    label="Prediction",
                    value=model_pred["verdict"],
                    delta=f"{model_pred['confidence']:.1f}% confidence"
                )

                if fact_check:
                    st.caption("🔎 Database matches:")
                    for fact, data in fact_check.items():
                        st.write(f"- {fact}: {data}")

    with col2:
        with st.container():
            st.subheader("🧠 Expert Verification")
            if "error" in llm_response:
                st.error(llm_response["error"])
            else:
                st.metric(
                    label="Verdict",
                    value=llm_response.get("verdict", "N/A"),
                    delta=llm_response.get("confidence", "")
                )

                if llm_response.get("errors", "None") != "None":
                    st.warning(f"🚨 Errors: {llm_response['errors']}")
                if llm_response.get("correction", "None") != "None":
                    st.success(f"📝 Correction: {llm_response['correction']}")
                if llm_response.get("sources"):
                    st.caption(f"📚 Sources: {llm_response['sources']}")
                if llm_response.get("rationale"):
                    st.caption(f"💡 Rationale: {llm_response['rationale']}")

    # Discrepancy analysis
    if model_pred["verdict"] and "verdict" in llm_response:
        st.divider()
        if model_pred["verdict"] != llm_response["verdict"]:
            st.error("⚠️ Critical Discrepancy Detected", icon="⚠️")
            cols = st.columns(2)
            cols[0].write(f"**Model says:** {model_pred['verdict']}")
            cols[1].write(f"**Expert says:** {llm_response['verdict']}")
            st.write("**Recommendation:** Verify with primary sources")
        else:
            st.success("✅ Verification Consensus", icon="✅")


# UI Layout
st.title("🔍 Fake News Detector Pro+")
st.caption("Advanced fact verification with AI model + expert analysis + fact database")

tab1, tab2 = st.tabs(["Single Claim", "Batch Analysis"])

with tab1:
    claim = st.text_area(
        "Enter claim to verify:",
        placeholder="e.g. 'The Apollo 11 mission landed humans on the Moon in 1969'",
        height=120,
        key="single_claim"
    )

    if st.button("Verify Claim", type="primary", key="verify_single"):
        try:
            claim = preprocess_claim(claim)
            fact_check = check_against_database(claim)

            with st.spinner("Running advanced verification..."):
                model_pred = predict_deberta(claim)
                llm_response = llm_verification(claim, model_pred, fact_check)
                display_result(claim, model_pred, llm_response, fact_check)

        except Exception as e:
            st.error(f"Verification failed: {str(e)}")

with tab2:
    st.warning("Batch processing may take several minutes for thorough verification")
    batch_claims = st.text_area(
        "Enter multiple claims (one per line):",
        height=200,
        key="batch_claims"
    )

    if st.button("Verify Batch", type="secondary", key="verify_batch"):
        try:
            claims = [preprocess_claim(c) for c in batch_claims.split('\n') if preprocess_claim(c)]
            progress_bar = st.progress(0)
            results = []

            for i, claim in enumerate(claims):
                progress_bar.progress((i + 1) / len(claims))
                fact_check = check_against_database(claim)
                model_pred = predict_deberta(claim)
                llm_response = llm_verification(claim, model_pred, fact_check)
                results.append((claim, model_pred, llm_response, fact_check))

            progress_bar.empty()

            for claim, model_pred, llm_response, fact_check in results:
                with st.expander(f"Claim: {claim[:60]}...", expanded=False):
                    display_result(claim, model_pred, llm_response, fact_check)

        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")

# Sample claims and database section
with st.expander("📋 Sample Claims & Database"):
    st.write("**Sample True Claims:**")
    st.code("""- The Apollo 11 mission landed humans on the Moon in 1969
- Water boils at 100°C at sea level
- COVID-19 vaccines underwent clinical trials""")

    st.write("**Sample False Claims:**")
    st.code("""- The Moon landing was filmed in a Hollywood studio
- Drinking bleach cures COVID-19
- Earth is flat""")

    st.write("**Current Fact Database:**")
    st.json(FACT_DATABASE)