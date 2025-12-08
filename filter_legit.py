import pandas as pd
import requests
import json
from tqdm import tqdm
import time
from requests.exceptions import Timeout, RequestException
# --- CẤU HÌNH ---

INPUT_FILES = ["content1", "content2", "content3","content4"]
MODEL = "gemma2:9b"


def get_legitimacy_prompt(text):
    return f"""
    Bạn là chuyên gia thẩm định tin tuyển dụng tại Việt Nam.
    Nhiệm vụ: Phân loại tin là UY TÍN (1) hoặc LỪA ĐẢO (0).

    TƯ DUY ĐÁNH GIÁ:
    - Ưu tiên gán nhãn 1 (UY TÍN) để không bỏ sót cơ hội việc làm.
    - Mức lương ở Việt Nam rất đa dạng, hãy nới lỏng tiêu chuẩn về lương.
    - ĐẶC BIỆT CHÚ Ý: Phân biệt rõ "Việc công ty/quán xá" (Cần địa chỉ) và "Việc gia đình/cá nhân" (Có thể thiếu địa chỉ).

    Văn bản: "{text}"

    --- ⛔ BỘ QUY TẮC ĐÁNH RỚT (LABEL 0 - SCAM) ---
    Nếu gặp bất kỳ dấu hiệu nào dưới đây -> LABEL 0 ngay lập tức:

    1. VIỆC NHẸ LƯƠNG TRÊN TRỜI:
       - Các việc online: Trực page, chốt đơn, đánh máy, nghe nhạc, xem video, like dạo.
       - Mức lương vô lý: > 300k-500k/ngày hoặc > 10 triệu/tháng cho việc KHÔNG CẦN KINH NGHIỆM.
       - Từ khóa mồi chài: "Không cọc không phí", "ib nhận việc ngay", "việc làm tại nhà cho mẹ bỉm".
       - Xưng hô thiếu chuyên nghiệp: "Chị cần..." (Ví dụ: "Chị cần 5 bạn...", "Chị đang cần gấp..."). Đây thường là văn phong của bọn tuyển sỉ/đa cấp/lừa đảo, trừ khi là tin tìm giúp việc gia đình thật sự.

    2. GIA CÔNG / THỦ CÔNG TẠI NHÀ:
       - Xâu hạt, xâu vòng, thêu tranh, gấp phong bì, bóc tỏi, nhặt yến.
       - Đây là chiêu trò lừa đảo cọc phổ biến nhất -> 0 tuyệt đối.

    3. DẪN DẮT QUA APP / TELEGRAM:
       - Yêu cầu tải App, kết bạn Telegram/Zalo để nhận lương.
       - Văn phong dùng quá nhiều icon tiền bạc (💰💸), hối thúc "chỉ còn 2 suất".

    4. LỪA ĐẢO ĐA CẤP:
       - Tuyển nhân viên kinh doanh/đối tác lương cứng 10-15tr không cần kinh nghiệm.
       - Địa chỉ chung chung: "Tại văn phòng", "Tại công ty" (Không có tên thương hiệu).

    --- ✅ TIÊU CHÍ CHẤP NHẬN (LABEL 1 - LEGIT) ---

    1. NHÓM KINH DOANH (Quán cafe, Shop, Nhà hàng, Cty):
       - BẮT BUỘC: Phải có ĐỊA CHỈ (Số nhà/Tên đường) HOẶC Tên Thương hiệu (Highlands, Winmart...).
       - Chấp nhận địa chỉ dân dã: "Quán cafe Xoài 44 phạm đình toái", "Bún đậu cô Ba".

    2. NHÓM GIA ĐÌNH/CÁ NHÂN (Giúp việc, Trông trẻ, Gia sư, Chăm sóc người già):
       - 🟢 NGOẠI LỆ ĐỊA CHỈ: CHẤP NHẬN không có số nhà cụ thể (vì lý do riêng tư).
       - 🔴 YÊU CẦU BẮT BUỘC: Phải có SĐT Liên Hệ rõ ràng + Mô tả công việc cụ thể.
       - Ví dụ OK: "Tìm cô giúp việc làm tại Cẩm Lệ, lương 8tr, LH 0905xxx" -> LABEL 1.
       - Ví dụ SCAM: "Cần người làm việc nhà gấp, lương cao, ib chị" (Không SĐT, văn phong mồi chài) -> LABEL 0.

    3. MỨC LƯƠNG & THỜI GIAN (Nới lỏng):
       - Part-time: 15k - 35k/giờ (Sinh viên).
       - Full-time: 5 - 18 triệu/tháng (Lao động phổ thông/Thợ/Sale/Đầu bếp).
       - Ca gãy, ca xoay: 2-6 tiếng/ngày là bình thường.
       - Không ghi lương (Thỏa thuận): Vẫn tính là LABEL 1 nếu các thông tin khác minh bạch.

    --- VÍ DỤ MINH HỌA ---
    VD1 (Lừa đảo - Xưng hô lạ + Việc nhẹ): "Chị cần 5 bạn trực page, 500k/ngày. Ib chị." -> {{ "label": 0, "risk_reason": "Việc nhẹ lương cao, xưng hô 'Chị cần' thiếu chuyên nghiệp, không rõ địa chỉ" }}
    VD2 (Lừa đảo - Gia công): "Tuyển chị em xâu hạt vòng về nhà làm." -> {{ "label": 0, "risk_reason": "Lừa đảo gia công tại nhà" }}
    VD3 (Uy tín - Quán xá): "Quán Nhậu Tự Do 234 Điện Biên Phủ tuyển phục vụ ca tối. Lương 25k/h." -> {{ "label": 1, "risk_reason": "Địa chỉ rõ ràng, lương thị trường" }}
    VD4 (Uy tín - Gia đình): "Gia đình cần tìm cô trông bé 6 tháng tại khu vực Hòa Xuân. Lương 9tr, bao ăn ở. Liên hệ: 0912.345.xxx." -> {{ "label": 1, "risk_reason": "Việc gia đình, chấp nhận thiếu số nhà vì có SĐT và mô tả rõ" }}
    VD5 (Uy tín - Gia sư): "Tìm gia sư dạy Toán lớp 5 khu vực Ngũ Hành Sơn. 150k/buổi. LH Zalo 09xx." -> {{ "label": 1, "risk_reason": "Việc gia sư cá nhân, có liên hệ rõ ràng" }}

    Hãy trả về JSON duy nhất:
    {{
        "label": 0 hoặc 1,
        "risk_reason": "Giải thích ngắn gọn lý do"
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
    print("--- BẮT ĐẦU BƯỚC 2: PHÂN TÍCH LỪA ĐẢO (CHỈ TRÊN TIN TUYỂN DỤNG) ---")

    for file_name in INPUT_FILES:
        # Đọc file kết quả từ Bước 1
        input_path = f"Data/step1/{file_name}_step1.csv"
        output_path = f"Data/step2/{file_name}_step2.csv"

        try:
            df = pd.read_csv(input_path)

            # CHỈ LỌC NHỮNG DÒNG LÀ TUYỂN DỤNG (is_recruitment == 1)
            # Những dòng = 0 (Quảng cáo/Rác) sẽ bị bỏ qua hoặc giữ nguyên không dán nhãn legit
            recruitment_df = df[df['is_recruitment'] == 1].copy()

            print(f"📂 Đang xử lý: {file_name} | Tìm thấy {len(recruitment_df)} tin tuyển dụng cần check.")

            col_name = 'content' if 'content' in recruitment_df.columns else recruitment_df.columns[0]

            results = []
            for idx, row in tqdm(recruitment_df.iterrows(), total=len(recruitment_df)):
                text = str(row[col_name])

                ai_res = call_ai_model(get_legitimacy_prompt(text))

                # Lưu kết quả vào row
                row['label'] = ai_res.get('label', 0)
                row['legit_reason'] = ai_res.get('risk_reason', '')

                results.append(row)

            # Xuất file kết quả (Chỉ chứa các bài tuyển dụng đã gán nhãn uy tín)
            final_df = pd.DataFrame(results)
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ Hoàn tất! File nhãn sạch được lưu tại: {output_path}")

        except FileNotFoundError:
            print(f"⚠️ Chưa chạy Bước 1 cho file {file_name} hoặc không tìm thấy file _step1.csv")


if __name__ == "__main__":
    main()