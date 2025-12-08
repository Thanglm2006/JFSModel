import pandas as pd
import requests
import json
import regex as re
from tqdm import tqdm
import time
from requests.exceptions import Timeout, RequestException
# --- CẤU HÌNH ---
INPUT_FILE = ["content3","content2","content1"]
MODEL = "gemma2:9b"


def check_icon_spam(text):
    """
    Kiểm tra xem bài đăng có bị 'bội thực' icon hay không.
    Trả về: (Score, Reason)
    """
    # 1. Danh sách các icon "nhạy cảm" thường dùng trong tin rác/lừa đảo
    spam_icons = ['🚨', '🆘', '📣', '💸', '💰', '💎', '💵', '👉', '👇', '🔥', '⚡', '✅', '❌']

    # 2. Đếm tổng số icon này trong bài
    spam_icon_count = 0
    for char in text:
        if char in spam_icons:
            spam_icon_count += 1

    # 3. Logic đánh giá
    # Nếu có trên 5 icon loại "tiền/báo động" -> Rất khả nghi
    if spam_icon_count >= 5:
        return 0, f"Spam tín hiệu: Chứa quá nhiều icon lôi kéo ({spam_icon_count} icon)"

    # 4. Kiểm tra mật độ icon bất thường (Ví dụ: cứ 10 ký tự lại có 1 icon)
    # Regex tìm tất cả emoji (cần thư viện regex)
    all_emojis = re.findall(r'\p{So}', text)
    if len(all_emojis) > 15 and (len(all_emojis) / len(text) > 0.15):  # Mật độ > 15%
        return 0, "Spam tín hiệu: Mật độ icon quá dày đặc, thiếu chuyên nghiệp"

    return None, None  # Không vi phạm


def analyze_post_optimized(text):
    # --- 1. KIỂM TRA HARD RULES (Cơ bản) ---
    icon_score, icon_reason = check_icon_spam(text)
    if icon_score is not None:
        return {
            "legit_score": 0,
            "risk_reason": f"[AUTO-FILTER] {icon_reason}",
            "salary": None,
            "position": None
        }

    # --- 2. PROMPT TƯ DUY KÉP (DUAL LOGIC) ---
    prompt = f"""
        Bạn là chuyên gia thẩm định tin tuyển dụng và phát hiện lừa đảo. Hãy phân tích bài đăng sau theo 3 lớp màng lọc nghiêm ngặt.

        Văn bản: "{text}"

        --- QUY TRÌNH THẨM ĐỊNH (PHẢI THỎA MÃN TẤT CẢ MỚI ĐƯỢC SCORE 1) ---

        1. MÀNG LỌC: ĐỊA ĐIỂM & DANH TÍNH (Như cũ)
           - Nhóm Kinh Doanh (Shop/Cty): BẮT BUỘC có Địa chỉ cụ thể HOẶC Tên Thương Hiệu rõ ràng (Highlands, Vinmart, KS Mường Thanh...). Chỉ ghi "Tại Đà Nẵng" -> LOẠI.
           - Nhóm Gia Đình (Giúp việc/Gia sư): CHẤP NHẬN không địa chỉ, NHƯNG phải có SĐT/Zalo + Mô tả việc rõ.

        2. MÀNG LỌC: TÍNH KHẢ THI VỀ LƯƠNG (QUAN TRỌNG)
           - Nguyên tắc thị trường: Lao động phổ thông (không bằng cấp) lương 15k-30k/giờ hoặc 5-8tr/tháng.
           - DẤU HIỆU LỪA ĐẢO (SCORE 0): 
             + Việc nhẹ lương trên trời (Gõ văn bản, xâu hạt, like dạo... mà lương 300k-500k/ngày hoặc 10tr/tháng).
             + Công việc đơn giản nhưng thu nhập >15tr/tháng không yêu cầu kinh nghiệm.

        3. MÀNG LỌC: TÍNH THỰC TẾ CÔNG VIỆC
           - DẤU HIỆU LỪA ĐẢO/RÁC (SCORE 0):
             + Các việc làm thủ công mang về nhà (xâu vòng, thêu tranh, gấp phong bì) -> 99% lừa đảo cọc.
             + Tuyển CTV chốt đơn, làm nhiệm vụ online, xem video kiếm tiền.
             + Tin tìm người yêu, Sugar Baby, kết bạn tâm sự (Nhóm C).
             + Tin cho vay vốn, cầm đồ, bán sim, bán đất.

        --- VÍ DỤ MẪU (FEW-SHOT) ---
        VD1 (Uy tín): "Highlands Coffee 123 Nguyễn Văn Linh tuyển phục vụ, 20k/h." -> Score: 1 (Địa chỉ rõ + Lương hợp lý).
        VD2 (Lừa đảo - Lương vô lý): "Tuyển nhân viên trực page tại nhà, lương 500k/ngày, không cần kinh nghiệm." -> Score: 0 (Lương quá cao so với việc nhẹ).
        VD3 (Lừa đảo - Việc rác): "Cần 50 chị em nhận hạt về xâu, công 3tr/tuần." -> Score: 0 (Lừa đảo gia công).
        VD4 (Gia đình - Uy tín): "Tìm cô giúp việc nhà Quận 7, lương 8tr, bao ăn ở. LH 0905xxx." -> Score: 1 (Lương 8tr bao ăn ở là giá thị trường hợp lý).
        VD5 (Rác): "Anh độc thân vui tính cần tìm bạn nữ đi cafe tâm sự, chu cấp 10tr." -> Score: 0 (Spam/Sugar baby).
        VD6 (Địa chỉ ảo): "Tuyển nhân viên kho, Lương 15tr, Địa chỉ: Khu vực Hải Châu." -> Score: 0 (Lương cao bất thường cho kho + Địa chỉ chung chung).
        VD6 (uy tín): "Địa điểm: 279 Nguyễn Tri Phương" -> Score: 1 (số + địa chỉ).
        VD7 (uy tín): "Cf 89 257 tô hiệu hoà minh liên chiểu cần tuyển 1 nv nữ ca sáng." -> Score: 1 (Có số + địa chỉ).
        Hãy trả về JSON duy nhất:
        {{  
            "category": "RECRUITMENT" hoặc "SPAM",
            "legit_score": 0 hoặc 1,
            "risk_reason": "Giải thích ngắn gọn lý do (VD: Lương 500k/ngày là vô lý cho việc trực page / Việc xâu hạt là lừa đảo / Địa chỉ và lương hợp lý...)",
            "salary": "Trích xuất mức lương hoặc null",
            "position": "Trích xuất vị trí hoặc null"
        }}
        """
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
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result_json = response.json()
            if 'response' in result_json:
                return json.loads(result_json['response'])

        except Timeout:
            print(f"⚠️ Timeout (Lần {attempt + 1})")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(1)

    return None

def main():
    print(f"🚀 Đang chạy model: {MODEL} với Strategy Few-Shot & Hard-Rules...")

    for file_name in INPUT_FILE:
        full_path ="Data/"+ file_name + ".csv"
        try:
            df = pd.read_csv(full_path)
            df_process = df
            print(f"📂 Đã tải: {full_path} | Cần xử lý: {len(df_process)} dòng")
        except FileNotFoundError:
            continue

        col_name = 'content' if 'content' in df.columns else df.columns[0]
        results = []

        for index, row in tqdm(df_process.iterrows(), total=df_process.shape[0], desc=f"Xử lý {file_name}"):
            text = str(row[col_name])

            # Lọc độ dài (tăng lên 20 ký tự để tránh tin rác quá ngắn)
            if len(text) < 20:
                continue

            ai_data = analyze_post_optimized(text)

            if ai_data:
                row_data = row.to_dict()

                # Cập nhật dữ liệu
                row_data['ai_category'] = ai_data.get('category', 'UNKNOWN')
                row_data['label'] = ai_data.get('legit_score', 0)
                row_data['ai_reason'] = ai_data.get('risk_reason', '')
                row_data['extracted_salary'] = ai_data.get('salary', '')
                row_data['extracted_pos'] = ai_data.get('position', '')

                results.append(row_data)

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            recruitment_df = result_df
            final_df = recruitment_df.sort_values(by='label', ascending=False)

            output_path ="Data/"+  file_name + "_labeled.csv"
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Hoàn tất: {output_path}")


if __name__ == "__main__":
    main()