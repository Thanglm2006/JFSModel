import re
import torch
import emoji
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "./my_scam_model"

app = FastAPI(title="Scam Detection API")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔄 Loading model from {MODEL_PATH} on {device}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you upload the model? If using HF Hub, make sure the repo ID is correct.")


# --- PREPROCESSING FUNCTIONS ---
def normalize_text(text):
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Normalize time
    text = re.sub(r'[\/\\]\s*(\d*h|giờ|tiếng)', ' một giờ ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(ngày|day)', ' một ngày ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(tháng|month)', ' một tháng ', text, flags=re.IGNORECASE)

    # Normalize money
    text = re.sub(r'\b(\d+)\s*(k|ka)\b', r'\1 nghìn', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*(tr|triệu|củ)\b', r'\1 triệu', text, flags=re.IGNORECASE)

    # Demojize
    return emoji.demojize(text, language='alias')


# --- API DATA MODELS ---
class ScamRequest(BaseModel):
    text: str


# --- API ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "online", "device": device}


@app.post("/predict")
def predict_scam(request: ScamRequest):
    # 1. Preprocess
    clean_text = normalize_text(request.text)

    # 2. Tokenize
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    # 3. Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Process Results
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    pred_label_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label_idx].item()

    # Map label (0/1) to string
    label_map = {0: "SCAM", 1: "LEGIT"}
    readable_label = label_map.get(pred_label_idx, "UNKNOWN")

    return {
        "original_text": request.text,
        "normalized_text": clean_text,
        "prediction": readable_label,  # "SCAM" or "LEGIT"
        "confidence": round(confidence, 4),
        "scores": {
            "scam": round(probs[0][0].item(), 4),
            "legit": round(probs[0][1].item(), 4)
        }
    }