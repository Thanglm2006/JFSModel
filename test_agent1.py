import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import emoji

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa Model 1 (Model phân loại Rác/Tuyển dụng)
MODEL_PATH = "./models/step1_mdeberta"  # Sửa đường dẫn này nếu bạn lưu tên khác

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# 1. LOAD MODEL
try:
    loaded_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    loaded_model.to(device)
    loaded_model.eval()
    print(f"✅ Đã load Model 1 thành công từ: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Lỗi: Không tìm thấy model tại '{MODEL_PATH}'.\nHãy kiểm tra lại đường dẫn.")
    exit()


def normalize_text(text):
    # Hàm chuẩn hóa giống hệt lúc train để đảm bảo model hiểu đúng
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Chuẩn hóa thời gian/lương
    text = re.sub(r'[\/\\]\s*(\d*h|giờ|tiếng)', ' một giờ ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(ngày|day)', ' một ngày ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(tháng|month)', ' một tháng ', text, flags=re.IGNORECASE)

    # Chuẩn hóa tiền
    text = re.sub(r'\b(\d+)\s*(k|ka|xu)\b', r'\1 nghìn', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*(tr|triệu|củ|m)\b', r'\1 triệu', text, flags=re.IGNORECASE)

    # Demojize
    return emoji.demojize(text, language='alias')


def predict_is_job(text):
    # Chuẩn hóa trước khi đưa vào model
    clean_text = normalize_text(text)

    inputs = loaded_tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)  # Chuyển sang xác suất %

    pred_label_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_label_idx].item()

    # --- NHÃN CỦA MODEL 1 ---
    # 0: Rác, Quảng cáo, Tìm việc
    # 1: Bài Tuyển dụng (Kể cả lừa đảo)
    label_map = {
        0: "🗑️ RÁC/SPAM/TÌM VIỆC (NON-JOB)",
        1: "📢 BÀI TUYỂN DỤNG (JOB)"
    }

    return label_map[pred_label_idx], confidence, probs[0].tolist()


# --- DỮ LIỆU TEST ---
# Bao gồm đủ các trường hợp để kiểm tra độ thông minh của model
test_texts = [
    # Case 1: Tuyển dụng uy tín (Mong đợi: JOB)
    "Highlands Coffee tuyển nhân viên phục vụ, lương 25k/h, làm tại Hải Châu.",
    "cần nhân viên phục vụ",

    # Case 2: Tuyển dụng lừa đảo (Mong đợi: JOB - Vì model này chỉ lọc rác, model 2 mới check scam)
    "Tuyển nhân viên xâu hạt tại nhà, lương 500k/ngày, không cần cọc.",

    # Case 3: Người tìm việc (Mong đợi: NON-JOB)
    "Em là sinh viên năm nhất, cần tìm việc làm thêm ca tối ạ. Ai có ib em với.",
    "em 2k3 đang kiếm công việc ca chiều ạ!",
    "em tìm cv phụ hồ",
    # Case 4: Quảng cáo bán hàng (Mong đợi: NON-JOB)
    "Thanh lý lô quần áo giá rẻ, ship toàn quốc. Mại dô mại dô 📣📣",

    # Case 5: Spam tài chính/Cho vay (Mong đợi: NON-JOB)
    "Hỗ trợ vay vốn sinh viên lãi suất thấp, giải ngân trong ngày.",

    # Case 6: Tin rác/Tâm sự (Mong đợi: NON-JOB)
    "Buồn quá có ai đi cafe nói chuyện cho vui không ạ?",
    "....",
    "okok"
]

print("\n" + "=" * 50)
print("--- KẾT QUẢ TEST MODEL 1 (FILTER) ---")
print("=" * 50)

for t in test_texts:
    label, conf, all_probs = predict_is_job(t)

    print(f"📝 Text: {t}")
    print(f"🎯 Kết quả: {label}")
    print(f"📊 Độ tin cậy: {conf:.2%}")
    print(f"📉 Chi tiết: [Rác: {all_probs[0]:.2%}, Tuyển dụng: {all_probs[1]:.2%}]")
    print("-" * 50)