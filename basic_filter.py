import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# --- CẤU HÌNH ---
INPUT_FILE = ["content1", "content2", "content3", "content4"]
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
BATCH_SIZE = 64  # RTX 3080 10GB/12GB cân tốt batch 64 hoặc 128

# Kiểm tra GPU
device = 0 if torch.cuda.is_available() else -1
print(f"⚡ Đang chạy trên: {'GPU (RTX 3080)' if device == 0 else 'CPU'}")

# Khởi tạo Pipeline
# Model này khoảng 2.2GB, load vào VRAM rất nhẹ
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=device
)


def process_files():
    for file_name in INPUT_FILE:
        full_path = "Data/" + file_name + ".csv"
        try:
            print(f"\n📂 Đang đọc file: {full_path}")
            df = pd.read_csv(full_path)

            # Reset index để đảm bảo đồng bộ
            df_process = df.copy().reset_index(drop=True)

            # Tự động tìm cột nội dung
            col_name = 'content' if 'content' in df_process.columns else df_process.columns[0]

            # Xử lý text: Chuyển về string, fillna
            # Quan trọng: Cắt bớt nếu quá dài để tránh lỗi model (model này giới hạn token)
            all_texts = df_process[col_name].fillna("").astype(str).apply(lambda x: x[:2000]).tolist()

            print(f"🚀 Đang phân loại {len(all_texts)} dòng...")

            # --- CẤU HÌNH NHÃN (LABELS) ---
            # Mẹo: Dùng từ khóa mô tả hành động để model dễ bắt bài Mixue
            candidate_labels = [
                "tin tuyển dụng tìm nhân viên",  # Nhãn mục tiêu
                "người tìm việc làm",  # Nhãn Job Seeker
                "quảng cáo rao vặt bán hàng",  # Nhãn Spam
                "spam rác xổ số tài chính"  # Nhãn Rác hẳn
            ]

            # --- CHẠY BATCH ---
            # hypothesis_template cực quan trọng cho tiếng Việt
            results = classifier(
                all_texts,
                candidate_labels,
                hypothesis_template="Bài viết này là về {}.",  # Giúp model hiểu ngữ cảnh
                multi_label=False,
                batch_size=BATCH_SIZE
            )

            # --- XỬ LÝ KẾT QUẢ ---
            final_categories = []
            final_scores = []

            for res in results:
                top_label = res['labels'][0]
                score = res['scores'][0]

                # Mapping về code ngắn gọn
                if top_label == "tin tuyển dụng tìm nhân viên":
                    cat = "RECRUITMENT"
                elif top_label == "người tìm việc làm":
                    cat = "JOB_SEEKER"
                else:
                    cat = "SPAM"

                final_categories.append(cat)
                final_scores.append(score)

            # Gán vào DF
            df_process['zs_category'] = final_categories
            df_process['zs_score'] = final_scores

            # --- LƯU KẾT QUẢ ---
            # Lưu file gốc kèm nhãn để kiểm tra
            output_full = "Data/" + file_name + "_labeled_roberta.csv"
            df_process.to_csv(output_full, index=False, encoding='utf-8-sig')

            # Thống kê
            n_recruit = len(df_process[df_process['zs_category'] == 'RECRUITMENT'])
            print(f"✅ Hoàn tất. Tìm thấy {n_recruit} bài tuyển dụng.")
            print(f"💾 Đã lưu: {output_full}")

        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {full_path}")
        except Exception as e:
            print(f"⚠️ Lỗi: {e}")


if __name__ == "__main__":
    process_files()