import pandas as pd
import requests
import json
from tqdm import tqdm
import time
from requests.exceptions import Timeout, RequestException

# --- CẤU HÌNH ---
INPUT_FILES = ["content1", "content2", "content3","content4"]  # Tên file gốc (không đuôi .csv)
MODEL = "gemma2:9b"


def get_category_prompt(text):
    return f"""
    Bạn là chuyên gia phân loại dữ liệu văn bản tiếng Việt cho nhóm "Tìm Việc Làm".
    Nhiệm vụ: Xác định xem văn bản dưới đây có phải là TIN TUYỂN DỤNG VIỆC LÀM hay không.

    Văn bản: "{text}"

    --- HƯỚNG DẪN GÁN NHÃN ---
    Gán nhãn "is_recruitment": 0 (KHÔNG PHẢI) hoặc 1 (PHẢI) dựa trên quy tắc sau:

    TRƯỜNG HỢP LÀ 0 (NON-RECRUITMENT / SPAM / AD):
    1. QUẢNG CÁO BÁN HÀNG & DỊCH VỤ:
       - Bán sản phẩm: Quần áo, giày dép, sim số, đồ ăn. (Từ khóa: "Giá sỉ", "100k/áo", "Freeship", "Thanh lý").
       - Dịch vụ tài chính: Cho vay, cầm đồ, rút ví trả sau.
       - Dịch vụ Marketing: Tăng like, chạy quảng cáo, nhận in ấn, thiết kế logo.
    2. TIN CÁ NHÂN & XÃ HỘI (User requested):
       - Tìm người yêu, tìm bạn đời, tìm FWB/ONS, Sugar Baby.
       - Tìm đồ thất lạc, pass đồ cũ, tìm phòng trọ (người thuê tìm phòng).

    TRƯỜNG HỢP LÀ 1 (IS_RECRUITMENT):
    1. TÌM NGƯỜI LÀM VIỆC:
       - Chứa từ khóa: "Tuyển nhân viên", "Cần tìm người làm", "Việc làm", "Tuyển gấp".
       - Mô tả vị trí: Phục vụ, bán hàng, bảo vệ, kế toán, tài xế, gia sư, giúp việc.
       - Tuyển Cộng tác viên (CTV) bán hàng/kinh doanh (Dù có thể là lừa đảo nhưng bản chất vẫn là tin tuyển dụng).
        - Mục đích bài đăng là tìm CON NGƯỜI để làm việc (Bất kể việc lớn hay nhỏ, chính thức hay làm thêm).
        - Chấp nhận các từ ngữ dân dã/sinh viên: "Cần 1 bạn", "Tìm người phụ", "Phụ bán", "Trông coi", "Giữ xe", "Cô giúp việc".
        - QUAN TRỌNG: Phân biệt "Bán hàng" (Spam) và "Tuyển người bán hàng" (Tuyển dụng).
          + "Bán bánh mì ngon lắm" -> 0 (Quảng cáo).
          + "Cần bạn phụ bán bánh mì" -> 1 (Tuyển dụng).
    --- VÍ DỤ MINH HỌA (FEW-SHOT) ---
    VD1 (Bán hàng): "Full in kelme đội cb giáng sinh, giá sx chỉ 100K/ÁO, hỗ trợ logo." 
       -> {{"is_recruitment": 0, "reason": "Tin quảng cáo bán áo và in ấn"}}

    VD2 (Tìm bạn/Tình cảm): "Nam 30t độc thân vui tính cần tìm bạn nữ tâm sự, đi cafe cuối tuần."
       -> {{"is_recruitment": 0, "reason": "Tin tìm bạn hẹn hò/tâm sự cá nhân"}}

    VD3 (Dịch vụ): "Hỗ trợ rút tiền ví trả sau momo, kredivo phí thấp."
       -> {{"is_recruitment": 0, "reason": "Quảng cáo dịch vụ tài chính"}}

    VD4 (Tuyển dụng): "Cần tuyển 2 bạn phục vụ cafe ca sáng, lương 20k/h."
       -> {{"is_recruitment": 1, "reason": "Tin tuyển dụng nhân viên phục vụ"}}
    VD5 (Tuyển dụng - Case sinh viên): "Cần 1 bạn sinh viên phụ bán bánh buổi sáng 6h15-7h30 trường tiểu học Lê Văn Hiến."
       -> {{"is_recruitment": 1, "reason": "Tìm người phụ bán hàng (Việc làm thêm)"}}

    Hãy trả về JSON duy nhất:
    {{
        "is_recruitment": 0 hoặc 1,
        "reason": "Giải thích ngắn gọn"
    }}
    """


def call_ai_model(prompt):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0.1, "num_ctx": 4096}
                },
                timeout=30  # 30 seconds
            )
            response.raise_for_status()  # Optional: raise for 4xx/5xx
            return json.loads(response.json()['response'])

        except Timeout:
            print(f"[{attempt + 1}/{max_retries}] Timeout sau 30s, đang thử lại...")
            time.sleep(3)
            continue

        except RequestException as e:
            print(f"[{attempt + 1}/{max_retries}] Lỗi kết nối Ollama: {e}")
            time.sleep(3)
            continue

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Lỗi parse JSON từ Ollama: {e}")
            print("Response raw:", response.text[:500] if 'response' in locals() else "No response")
            time.sleep(1)
            continue

        except Exception as e:
            print(f"Lỗi không xác định: {e}")
            time.sleep(1)
            continue

    # Sau 3 lần thất bại
    print("Đã thử 3 lần nhưng đều thất bại → Gán nhãn mặc định là SCAM (an toàn hơn)")
    return {"label": 0, "risk_reason": "Lỗi kết nối/model timeout sau 3 lần thử"}


def main():
    print("--- BẮT ĐẦU BƯỚC 1: PHÂN LOẠI TIN TUYỂN DỤNG ---")

    for file_name in INPUT_FILES:
        input_path = f"Data/{file_name}.csv"
        output_path = f"Data/{file_name}_step1.csv"

        try:
            df = pd.read_csv(input_path)
            col_name = 'content' if 'content' in df.columns else df.columns[0]
            print(f"📂 Đang xử lý: {file_name} ({len(df)} dòng)")

            results = []
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                text = str(row[col_name])
                if len(text) < 15:  # Bỏ qua tin quá ngắn
                    row['is_recruitment'] = 0
                    row['cat_reason'] = "Too short"
                else:
                    ai_res = call_ai_model(get_category_prompt(text))
                    row['is_recruitment'] = ai_res.get('is_recruitment', 0)
                    row['cat_reason'] = ai_res.get('reason', '')

                results.append(row)

            pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Đã lưu: {output_path}")

        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {input_path}")


if __name__ == "__main__":
    main()