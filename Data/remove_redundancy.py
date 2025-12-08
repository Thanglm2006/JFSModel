import pandas as pd
import torch
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CẤU HÌNH ---
INPUT_FILES = ["raw/facebook", "raw/data_viet"]  # Tên file của bạn
TARGET_COLUMN = 'text'
THRESHOLD = 0.90  # Ngưỡng trùng lặp 90%
BATCH_SIZE = 2000  # Giảm nếu bị lỗi Memory

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang chạy trên thiết bị: {device}")


def process_gpu_duplicates():
    # 1. ĐỌC DỮ LIỆU VÀ GỘP CỘT
    all_data = []
    print("--- Giai đoạn 1: Đọc và tiền xử lý dữ liệu ---")

    for filename in INPUT_FILES:
        file_path = f"{filename}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)

                # --- LOGIC GỘP 2 CỘT ĐẦU TIÊN ---
                # Kiểm tra xem file có ít nhất 2 cột không
                if len(df.columns) >= 2:
                    # Lấy nội dung 2 cột đầu (index 0 và 1)
                    # .fillna('') để xử lý ô trống thành chuỗi rỗng
                    # .astype(str) để đảm bảo là chuỗi
                    col_1 = df.iloc[:, 0].fillna('').astype(str)
                    col_2 = df.iloc[:, 1].fillna('').astype(str)

                    # Gộp lại thành cột 'text', ngăn cách bởi xuống dòng
                    df[TARGET_COLUMN] = col_1 + "\n" + col_2
                    print(f"   ℹ️ Đã gộp cột '{df.columns[0]}' và '{df.columns[1]}' thành 'text'.")
                else:
                    print(f"   ⚠️ File {filename} có ít hơn 2 cột, bỏ qua bước gộp.")
                    # Nếu không gộp được thì phải đảm bảo có cột text, nếu không tạo rỗng
                    if TARGET_COLUMN not in df.columns:
                        df[TARGET_COLUMN] = ""
                # ------------------------------------

                df['__source_file__'] = filename
                df['__original_index__'] = df.index

                # Làm sạch dữ liệu lần cuối để tránh lỗi
                df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("").astype(str)

                all_data.append(df)
                print(f"✅ Đã tải: {filename}.csv ({len(df)} dòng)")
            except Exception as e:
                print(f"❌ Lỗi đọc file {filename}: {e}")

    if not all_data:
        print("Không có dữ liệu đầu vào.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    texts = combined_df[TARGET_COLUMN].tolist()

    print(f"\n--- Giai đoạn 2: Vector hóa dữ liệu ({len(texts)} dòng) ---")

    # Vector hóa
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)

    print("--- Giai đoạn 3: Tính toán tương đồng trên GPU ---")

    # Chuyển đổi sang PyTorch Sparse Tensor
    coo = tfidf_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data)
    shape = coo.shape

    sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

    try:
        full_dense = sparse_tensor.to_dense()
    except RuntimeError as e:
        print(f"❌ Lỗi bộ nhớ GPU: {e}")
        return

    drop_indices = set()
    n_samples = full_dense.shape[0]

    print(f"Đang quét trùng lặp trên {n_samples} dòng...")

    for i in range(0, n_samples, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n_samples)
        batch_vectors = full_dense[i:end]
        sim_matrix = torch.mm(batch_vectors, full_dense.T)
        sim_vals_cpu = sim_matrix.cpu().numpy()

        for local_idx in range(end - i):
            global_idx = i + local_idx
            if global_idx in drop_indices: continue

            row_sims = sim_vals_cpu[local_idx]
            previous_matches = np.where(row_sims[:global_idx] >= THRESHOLD)[0]

            is_duplicate = False
            for match_idx in previous_matches:
                if match_idx not in drop_indices:
                    is_duplicate = True
                    break

            if is_duplicate:
                drop_indices.add(global_idx)

        print(f"✅ Đã xử lý xong batch {i}-{end}", end='\r')

    # 4. XUẤT FILE
    print(f"\n\n--- Giai đoạn 4: Xuất kết quả ---")

    all_indices = set(range(len(combined_df)))
    keep_indices = list(all_indices - drop_indices)
    keep_indices.sort()

    df_clean_global = combined_df.iloc[keep_indices]

    print(f"Tổng ban đầu: {len(combined_df)} | Sau khi lọc: {len(df_clean_global)}")
    print(f"Đã loại bỏ: {len(drop_indices)} dòng.")

    for filename in INPUT_FILES:
        df_part = df_clean_global[df_clean_global['__source_file__'] == filename].copy()

        # Xóa cột tạm
        cols_to_drop = ['__source_file__', '__original_index__']
        # Nếu bạn không muốn giữ lại cột 'text' gộp trong file kết quả, bỏ comment dòng dưới:
        # cols_to_drop.append('text')

        df_part = df_part.drop(columns=[c for c in cols_to_drop if c in df_part.columns])

        output_name = f"{filename}_filtered.csv"
        df_part.to_csv(output_name, index=False, encoding='utf-8-sig')
        print(f"📁 Đã lưu: {output_name} ({len(df_part)} dòng)")


if __name__ == "__main__":
    if torch.cuda.is_available():
        process_gpu_duplicates()
    else:
        print("❌ Cần GPU để chạy code này.")