import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import emoji
model_path = "models/step2_mdeberta"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on: {device}")

loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
loaded_model.to(device)
loaded_model.eval()

print("Model loaded")


def predict_scam(text):
    # Tokenize input
    inputs = loaded_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    logits = outputs.logits

    # use softmax to turn res to probabilities
    probs = F.softmax(logits, dim=-1)

    # take the highest probability
    pred_label_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label_idx].item()

    label_map = {0: "⚠️ LỪA ĐẢO (SCAM)", 1: "✅ UY TÍN (LEGIT)"}

    return label_map[pred_label_idx], confidence, probs[0].tolist()

def convert_emoji(text):
    if not isinstance(text, str): # Nếu không phải chuỗi (ví dụ là nan/float)
        return str(text)
    return emoji.demojize(text, language='alias')


def normalize_text(text):
    # 1. Kiểm tra an toàn trước tiên
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # LƯU Ý: Đã xóa dòng text = text.lower()

    # 2. Chuẩn hóa thời gian/lương (Thêm flags=re.IGNORECASE)
    # Bắt: /h, /H, /giờ, /Giờ...
    text = re.sub(r'[\/\\]\s*(\d*h|giờ|tiếng)', ' một giờ ', text, flags=re.IGNORECASE)

    # Bắt: /ngày, /Day, /Ngày...
    text = re.sub(r'[\/\\]\s*(ngày|day)', ' một ngày ', text, flags=re.IGNORECASE)

    # Bắt: /tháng, /Month...
    text = re.sub(r'[\/\\]\s*(tháng|month)', ' một tháng ', text, flags=re.IGNORECASE)

    # 3. Chuẩn hóa đơn vị tiền
    # Bắt: 100k, 100K, 100ka, 100KA
    text = re.sub(r'\b(\d+)\s*(k|ka)\b', r'\1 nghìn', text, flags=re.IGNORECASE)

    # Bắt: 5tr, 5TR, 5Tr, 5củ...
    text = re.sub(r'\b(\d+)\s*(tr|triệu|củ|m)\b', r'\1 triệu', text, flags=re.IGNORECASE)

    # 4. Demojize (Chuyển icon thành text :smile:)
    return emoji.demojize(text, language='alias')
test_texts = [
    "Tuyển dụng nhân viên nhập liệu tại nhà, không cần cọc, lương 500k/ngày, inbox nhận việc ngay 💰💰💰",
    "Công ty FPT Software tuyển dụng Kỹ sư cầu nối (BrSE), yêu cầu tiếng Nhật N2, kinh nghiệm 2 năm.",
        "🔥Quán cafe ông kẹ tuyển nhân viên phục vụ, lương 20k/h, ✅lịch làm: 7h-11h từ thứ 2 đến thứ 7.",
    """
    🥇Chỗ mình đang cần người phụ bán hàng các mặt hàng organic.
🥇Thời gian từ 8h đến 17h chiều
🥇Có thể làm cả ngày hay 1 buổi.
 - Thời gian: 
+ Sáng từ 8h - 11h30
+ Chiều từ 13h30 - 17h00
🥇Tuổi 
+Nam từ 20 tuổi - 30 tuổi
+Nữ từ 20 tuổi - 55 tuổi
Bạn nào có nhu cầu xin liên hệ 0932580161
"""
]

print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
for t in test_texts:
    label, conf, all_probs = predict_scam(normalize_text(t))
    print(f"📝 Text: {normalize_text(t)}")
    print(f"🎯 Result: {label}")
    print(f"📊 Accuracy: {conf:.2%}")
    print(f"📉 Probabilitíe: [Lừa đảo: {all_probs[0]:.2%}, Uy tín: {all_probs[1]:.2%}]")
    print("-" * 30)
