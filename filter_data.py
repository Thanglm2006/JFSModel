import pandas as pd
import requests
import json
import regex as re
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_FILE = ["content2","content3"]
MODEL = "qwen3:8b"


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
def pre_check_scam(text):
    text_lower = text.lower()

    # 1. Các từ khóa "báo động đỏ" (Scam/Đa cấp/Cờ bạc)
    scam_keywords = [
        "xâu hạt", "gấp phong bì", "gõ mã", "nhập liệu tại nhà",
        "việc nhẹ lương cao", "chốt đơn", "hoa hồng cao",
        "tài xỉu", "game bài", "kèo bóng", "kiếm 500k/ngày"
    ]

    # 2. Các từ khóa "ẩn danh" (thường là lừa đảo hoặc kém uy tín)
    anon_keywords = ["inbox", "ib riêng", "nhắn tin riêng", "ib mình", "không cọc"]

    for kw in scam_keywords:
        if kw in text_lower:
            return 0, f"Chứa từ khóa rủi ro cao: {kw}"

    # Nếu bài quá ngắn và đòi inbox -> Rủi ro
    if len(text) < 100 and any(kw in text_lower for kw in anon_keywords):
        return 0, "Bài viết ngắn và yêu cầu Inbox ẩn danh"

    return None, None  # Không vi phạm quy tắc cứng, chuyển cho AI


def analyze_post_optimized(text):
    # Kiểm tra Hard Rules trước
    icon_score, icon_reason = check_icon_spam(text)
    if icon_score is not None:
        return {
            "category": "SPAM",  # Hoặc RECRUITMENT nhưng score 0
            "legit_score": 0,
            "risk_reason": f"[AUTO-FILTER] {icon_reason}",
            "salary": None,
            "position": None
        }
    pre_score, pre_reason = pre_check_scam(text)
    if pre_score is not None:
        # Giả lập output giống AI để code phía sau chạy được
        return {
            "category": "RECRUITMENT",  # Tạm gán để lọc vào danh sách check
            "legit_score": 0,
            "risk_reason": f"[AUTO-FILTER] {pre_reason}",
            "salary": None,
            "position": None
        }

    # --- PROMPT FEW-SHOT (CUNG CẤP VÍ DỤ) ---
    prompt = f"""
    Bạn là một Chuyên gia Thẩm định Tin tuyển dụng khắt khe.
    Nhiệm vụ: Phân tích văn bản và trả về JSON.

    --- VÍ DỤ MẪU (HỌC THEO CÁCH ĐÁNH GIÁ NÀY) ---

    VD1 (Tin rác/Spam):
    Input: "Vay vốn sinh viên lãi suất thấp, giải ngân nhanh."
    Output: {{"category": "SPAM", "legit_score": 0, "risk_reason": "Quảng cáo dịch vụ tài chính, không phải tuyển dụng", "salary": null, "position": null}}

    VD2 (Tin Lừa đảo/Kém uy tín):
    Input: "Chị cần 2 bạn phụ bán hàng tại chỗ. Ai làm ib chị nhé. Lương 10tr."
    Output: {{"category": "RECRUITMENT", "legit_score": 0, "risk_reason": "Không có địa chỉ cụ thể, yêu cầu Inbox riêng, lương cao bất thường so với mô tả sơ sài", "salary": "10 triệu", "position": "Phụ bán hàng"}}

    VD3 (Tin Uy tín):
    Input: "Cafe Mộc 15 Lê Lợi, Đà Nẵng tuyển nhân viên phục vụ. Lương 25k/h. Ca sáng 7h-11h. LH: 0905xxx."
    Output: {{"category": "RECRUITMENT", "legit_score": 1, "risk_reason": "Địa chỉ rõ ràng (Số nhà + Tên đường), công việc cụ thể, mức lương hợp lý", "salary": "25k/h", "position": "Phục vụ"}}

    --- BÀI CẦN PHÂN TÍCH ---
    Văn bản: "{text}"

    --- YÊU CẦU LOGIC ---
    1. CATEGORY: RECRUITMENT (Tuyển người), JOB_SEEKER (Tìm việc), SPAM.
    2. LEGIT_SCORE (0 hoặc 1):
       - BẮT BUỘC PHẢI CÓ: Địa chỉ cụ thể (Số nhà/Tên đường/Tòa nhà) HOẶC Tên Thương Hiệu rõ ràng (Highlands, Vinmart...).
       - NẾU: Chỉ ghi "Khu vực Cẩm Lệ", "Tại Đà Nẵng" -> RỦI RO (Score 0).
       - NẾU: Yêu cầu "Ib/Inbox" mà không có SĐT/Địa chỉ -> RỦI RO (Score 0).

    Hãy trả về JSON duy nhất:
    """

    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,  # Giảm về 0 để logic chặt chẽ nhất
                "num_ctx": 4096,
                "top_p": 0.9
            }
        })
        return json.loads(response.json()['response'])
    except Exception as e:
        return None


def main():
    print(f"🚀 Đang chạy model: {MODEL} với Strategy Few-Shot & Hard-Rules...")

    for file_name in INPUT_FILE:
        full_path = file_name + ".csv"
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

        # ... (Phần lưu file giữ nguyên như code cũ) ...
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            recruitment_df = result_df[result_df['ai_category'] == 'RECRUITMENT']
            # Kết hợp lại với df_ok (tin đã chuẩn từ trước)
            final_df = recruitment_df.sort_values(by='label', ascending=False)

            output_path = file_name + "_labeled.csv"
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Hoàn tất: {output_path}")


if __name__ == "__main__":
    main()