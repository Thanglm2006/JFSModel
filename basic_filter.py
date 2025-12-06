import pandas as pd
import requests
import json
import math
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_FILE = ["content1_labeled", "content2_labeled", "content3_labeled", "data_content1_labeled"]
MODEL = "qwen:4b"  # Đảm bảo bạn đã pull model này
BATCH_SIZE = 30  # Kích thước lô


def extract_json_from_text(text):
    """
    Hàm phụ trợ: Cố gắng tìm và cắt chuỗi JSON hợp lệ từ phản hồi của AI
    nếu AI có lỡ chat thêm (VD: "Here is your json: [...]")
    """
    try:
        # Tìm vị trí bắt đầu của mảng JSON '[' và kết thúc ']'
        start = text.find('[')
        end = text.rfind(']') + 1

        if start != -1 and end != -1:
            json_str = text[start:end]
            return json.loads(json_str)
        return []
    except Exception:
        return []


def analyze_batch(texts_list, start_id_offset):
    """
    Xử lý một lô bài đăng.
    start_id_offset: ID bắt đầu để gán cho bài viết trong lô này (để mapping ngược lại).
    """
    if not texts_list:
        return {}

    # 1. Tạo chuỗi input với ID cụ thể cho từng bài
    # ID này chỉ dùng tạm trong prompt để AI biết bài nào là bài nào
    prompt_content = ""
    id_map = []  # Lưu danh sách ID trong lô này để kiểm tra sau

    for i, text in enumerate(texts_list):
        current_id = start_id_offset + i
        id_map.append(current_id)
        # Làm sạch text một chút để tránh phá vỡ prompt (xóa xuống dòng thừa)
        clean_text = str(text).replace('\n', ' ').replace('"', "'")[:500]  # Cắt 500 ký tự để tiết kiệm token
        prompt_content += f"ID_{current_id}: {clean_text}\n"

    # 2. Prompt Yêu cầu trả về ID
    prompt = f"""
    Bạn là AI phân loại tin tuyển dụng.

    DANH SÁCH BÀI ĐĂNG:
    {prompt_content}

    YÊU CẦU:
    - Phân loại từng bài đăng theo ID tương ứng.
    - Category chỉ chọn: "RECRUITMENT" (Tuyển dụng), "JOB_SEEKER" (Tìm việc), "SPAM_ADS" (Rác/QC).
    - Trả về kết quả dạng JSON Array. Bắt buộc phải giữ đúng ID đã cung cấp (Ví dụ: ID_{start_id_offset}).

    OUTPUT FORMAT (JSON ONLY):
    [
      {{"id": "ID_{start_id_offset}", "category": "RECRUITMENT"}},
      {{"id": "ID_{start_id_offset + 1}", "category": "SPAM_ADS"}}
    ]
    """

    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 4096
            },
            "keep_alive": "10m"
        })

        response_text = response.json()['response']

        # Dùng hàm trích xuất an toàn
        json_data = extract_json_from_text(response_text)

        # 3. CHUYỂN ĐỔI KẾT QUẢ VỀ DẠNG DICTIONARY {ID: CATEGORY}
        # Điều này giúp ta map chính xác 1-1, bất chấp thứ tự AI trả về
        result_map = {}
        for item in json_data:
            # Lấy số từ chuỗi "ID_123" -> 123
            try:
                raw_id = item.get('id', '')
                # Nếu AI trả về số nguyên (123) hoặc chuỗi ("ID_123")
                if isinstance(raw_id, int):
                    idx = raw_id
                else:
                    idx = int(raw_id.replace("ID_", ""))

                result_map[idx] = item.get('category', 'UNKNOWN')
            except ValueError:
                continue

        return result_map

    except Exception as e:
        print(f"⚠️ Lỗi xử lý lô: {e}")
        return {}


def main():
    print(f"🚀 Đang chạy model: {MODEL} | BATCH SIZE = {BATCH_SIZE}")

    for file_name in INPUT_FILE:
        full_path = file_name + ".csv"
        try:
            df = pd.read_csv(full_path)
            # Tạo bản sao và reset index để đảm bảo index chạy từ 0 -> n
            df_process = df.copy().reset_index(drop=True)
            print(f"📂 Đã tải: {full_path} | {len(df_process)} dòng")
        except FileNotFoundError:
            print(f"❌ Không tìm thấy: {full_path}")
            continue

        col_name = 'content' if 'content' in df.columns else df.columns[0]

        # Thêm cột kết quả mặc định
        df_process['ai_category'] = 'PENDING'

        all_texts = df_process[col_name].astype(str).tolist()
        num_batches = math.ceil(len(all_texts) / BATCH_SIZE)

        # --- VÒNG LẶP BATCH ---
        for i in tqdm(range(num_batches), desc=f"Xử lý {file_name}"):

            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(all_texts))

            current_batch_texts = all_texts[start_idx:end_idx]

            # GỌI HÀM XỬ LÝ (Truyền index bắt đầu để làm ID)
            # start_idx chính là ID của dòng đầu tiên trong lô này
            batch_results_map = analyze_batch(current_batch_texts, start_id_offset=start_idx)

            # CẬP NHẬT DATAFRAME DỰA TRÊN ID MAP
            # Duyệt qua các ID trong lô hiện tại
            for row_idx in range(start_idx, end_idx):
                # Nếu ID này có trong kết quả trả về của AI
                if row_idx in batch_results_map:
                    df_process.at[row_idx, 'ai_category'] = batch_results_map[row_idx]
                else:
                    # Nếu AI bỏ sót bài này, đánh dấu lỗi hoặc UNKNOWN
                    df_process.at[row_idx, 'ai_category'] = 'ERROR_MISSING'

        # --- LƯU KẾT QUẢ ---
        if not df_process.empty:
            # Lọc kết quả
            final_recruitment_df = df_process[df_process['ai_category'] == 'RECRUITMENT'].copy()

            print(f"\n✅ Hoàn thành file: {file_name}")
            print(f"- Tổng: {len(df_process)}")
            print(f"- Tuyển dụng: {len(final_recruitment_df)}")
            print(f"- Lỗi/Spam: {len(df_process) - len(final_recruitment_df)}")

            output_path = file_name + "_batched_classified.csv"
            final_recruitment_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"💾 Đã lưu: {output_path}\n")


if __name__ == "__main__":
    main()