import re
import torch
import emoji
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

# --- CẤU HÌNH ĐƯỜNG DẪN MODEL ---
# Hãy thay đổi đường dẫn này trỏ đến đúng thư mục model bạn đã train
MODEL_FILTER_PATH = "./models/step1/step1"  # Model 1: Lọc bài (0: Rác, 1: Tuyển dụng)
MODEL_SCAM_PATH = "./models/step2/step2"  # Model 2: Check Scam (0: Scam, 1: Legit)

# Khởi tạo App
app = FastAPI(
    title="JFS - Job Filtering System API",
    description="Hệ thống lọc tin tuyển dụng 2 bước: Lọc Rác -> Phát hiện Lừa đảo",
    version="2.0"
)

# Thiết lập Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Running on: {device}")

# --- BIẾN TOÀN CỤC ĐỂ LƯU MODEL ---
models = {}


# --- HÀM KHỞI TẠO (LOAD MODEL KHI START APP) ---
@app.on_event("startup")
async def load_models():
    print("🔄 Đang tải các model lên RAM/VRAM...")
    try:
        # 1. Load Model Filter (Bước 1)
        print(f"   - Loading Model 1 (Filter) from {MODEL_FILTER_PATH}...")
        models['filter_tokenizer'] = AutoTokenizer.from_pretrained(MODEL_FILTER_PATH)
        models['filter_model'] = AutoModelForSequenceClassification.from_pretrained(MODEL_FILTER_PATH)
        models['filter_model'].to(device).eval()

        # 2. Load Model Scam (Bước 2)
        print(f"   - Loading Model 2 (Scam) from {MODEL_SCAM_PATH}...")
        models['scam_tokenizer'] = AutoTokenizer.from_pretrained(MODEL_SCAM_PATH)
        models['scam_model'] = AutoModelForSequenceClassification.from_pretrained(MODEL_SCAM_PATH)
        models['scam_model'].to(device).eval()

        print("✅ Đã tải thành công cả 2 Model!")

    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi tải model: {e}")
        print("💡 Gợi ý: Kiểm tra lại đường dẫn folder model.")
        raise e


# --- HÀM XỬ LÝ TEXT (PREPROCESSING) ---
def normalize_text(text):
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Chuẩn hóa thời gian
    text = re.sub(r'[\/\\]\s*(\d*h|giờ|tiếng)', ' một giờ ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(ngày|day)', ' một ngày ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(tháng|month)', ' một tháng ', text, flags=re.IGNORECASE)

    # Chuẩn hóa tiền
    text = re.sub(r'\b(\d+)\s*(k|ka|xu)\b', r'\1 nghìn', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*(tr|triệu|củ)\b', r'\1 triệu', text, flags=re.IGNORECASE)

    # Demojize
    return emoji.demojize(text, language='alias')


# --- DATA MODELS (INPUT/OUTPUT) ---
class TextRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    final_decision: str  # NON_RECRUITMENT | SCAM | LEGIT
    step1_is_recruitment: int
    step1_confidence: float
    step2_is_legit: Optional[int] = None
    step2_confidence: Optional[float] = None
    normalized_text: str


# --- HÀM DỰ ĐOÁN CỐT LÕI ---
def predict_pipeline(text: str):
    clean_text = normalize_text(text)

    # --- BƯỚC 1: FILTER (Rác vs Tuyển dụng) ---
    tokenizer1 = models['filter_tokenizer']
    model1 = models['filter_model']

    inputs1 = tokenizer1(clean_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

    with torch.no_grad():
        outputs1 = model1(**inputs1)
        probs1 = F.softmax(outputs1.logits, dim=-1)

    # Model 1: 0 = Rác, 1 = Tuyển dụng
    is_recruitment_idx = torch.argmax(probs1, dim=1).item()
    is_recruitment_score = probs1[0][is_recruitment_idx].item()

    # Nếu Model 1 bảo KHÔNG PHẢI TUYỂN DỤNG (0) -> Dừng luôn
    if is_recruitment_idx == 0 and is_recruitment_score>=0.7:
        return {
            "final_decision": "NON_RECRUITMENT",
            "step1_is_recruitment": 0,
            "step1_confidence": round(is_recruitment_score, 4),
            "step2_is_legit": None,
            "step2_confidence": None,
            "normalized_text": clean_text
        }

    # --- BƯỚC 2: SCAM CHECK (Scam vs Uy tín) ---
    # Chỉ chạy khi Bước 1 là Tuyển dụng (1)
    tokenizer2 = models['scam_tokenizer']
    model2 = models['scam_model']

    inputs2 = tokenizer2(clean_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

    with torch.no_grad():
        outputs2 = model2(**inputs2)
        probs2 = F.softmax(outputs2.logits, dim=-1)

    # Model 2: 0 = Scam, 1 = Legit
    is_legit_idx = torch.argmax(probs2, dim=1).item()
    is_legit_score = probs2[0][is_legit_idx].item()

    final_label = "LEGIT" if is_legit_idx == 1 else "SCAM"

    return {
        "final_decision": final_label,
        "step1_is_recruitment": 1,
        "step1_confidence": round(is_recruitment_score, 4),
        "step2_is_legit": is_legit_idx == 1,
        "step2_confidence": round(is_legit_score, 4),
        "normalized_text": clean_text
    }


# --- ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "JFS System Ready", "device": device}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result = predict_pipeline(request.text)
    return result


# # --- CHẠY SERVER (NẾU CHẠY TRỰC TIẾP) ---
if __name__ == "__main__":
    import uvicorn

    # Chạy server tại localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)