import pandas as pd
import torch
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CẤU HÌNH ---
INPUT_FILES = ["facebook","data_viet"]
TARGET_COLUMN = 'text'
THRESHOLD = 0.90  # 90%
BATCH_SIZE = 2000  # Giảm nhẹ batch size để an toàn hơn

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang chạy trên thiết bị: {device}")


def process_gpu_duplicates():
    # 1. ĐỌC DỮ LIỆU
    all_data = []
    print("--- Giai đoạn 1: Đọc dữ liệu ---")
    for filename in INPUT_FILES:
        file_path = f"raw/{filename}.csv"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['__source_file__'] = filename
                df['__original_index__'] = df.index
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

    # Tinh chỉnh vectorizer để xử lý tiếng Việt và so khớp mờ tốt hơn
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)

    print("--- Giai đoạn 3: Tính toán tương đồng trên GPU ---")

    # --- PHẦN SỬA LỖI ---
    # Chuyển đổi định dạng Sparse Matrix từ Scikit-learn (COO) sang PyTorch
    coo = tfidf_matrix.tocoo()

    # Tạo indices (dạng [2, N]) và values
    indices = np.vstack((coo.row, coo.col))

    # Chuyển sang Tensor
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data)
    shape = coo.shape

    # Dùng sparse_coo_tensor (Hàm mới thay thế sparse_FloatTensor)
    sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)
    # --------------------

    # Chuyển sang Dense matrix để tính toán nhanh (Vì dữ liệu < 10.000 dòng nên RAM GPU chịu tốt)
    # Nếu bị lỗi Out Of Memory ở dòng này, hãy báo mình để đổi sang cách tính từng lô (batch)
    try:
        full_dense = sparse_tensor.to_dense()
    except RuntimeError as e:
        print(f"❌ Lỗi bộ nhớ GPU: {e}")
        print("💡 Giải pháp: Giảm dữ liệu hoặc chuyển sang CPU.")
        return

    drop_indices = set()
    n_samples = full_dense.shape[0]

    print(f"Đang quét trùng lặp trên {n_samples} dòng...")

    # Vòng lặp xử lý theo Batch
    for i in range(0, n_samples, BATCH_SIZE):
        end = min(i + BATCH_SIZE, n_samples)

        # Lấy batch hiện tại
        batch_vectors = full_dense[i:end]

        # Tính ma trận tương đồng: Batch x All
        sim_matrix = torch.mm(batch_vectors, full_dense.T)

        # Chuyển kết quả về CPU để xử lý logic (tránh thao tác index phức tạp trên GPU)
        sim_vals_cpu = sim_matrix.cpu().numpy()

        for local_idx in range(end - i):
            global_idx = i + local_idx

            if global_idx in drop_indices:
                continue

            # Lấy dòng tương đồng
            row_sims = sim_vals_cpu[local_idx]

            # Tìm các dòng TRƯỚC dòng hiện tại có độ giống > THRESHOLD
            # Chúng ta chỉ quan tâm [:global_idx] vì muốn giữ dòng xuất hiện trước, xóa dòng sau
            previous_matches = np.where(row_sims[:global_idx] >= THRESHOLD)[0]

            # Kiểm tra xem bản gốc của nó có bị xóa chưa?
            # Nếu bản gốc (dòng xuất hiện trước) vẫn còn -> Dòng này là thừa -> Xóa
            is_duplicate = False
            for match_idx in previous_matches:
                if match_idx not in drop_indices:
                    is_duplicate = True
                    break

            if is_duplicate:
                drop_indices.add(global_idx)

        print(f"✅ Đã xử lý xong batch {i}-{end}")

    # 4. LỌC VÀ XUẤT FILE
    print(f"\n--- Giai đoạn 4: Xuất kết quả ---")

    all_indices = set(range(len(combined_df)))
    keep_indices = list(all_indices - drop_indices)
    keep_indices.sort()

    df_clean_global = combined_df.iloc[keep_indices]

    print(f"Tổng ban đầu: {len(combined_df)} | Sau khi lọc: {len(df_clean_global)}")
    print(f"Đã loại bỏ: {len(drop_indices)} dòng.")

    for filename in INPUT_FILES:
        df_part = df_clean_global[df_clean_global['__source_file__'] == filename].copy()

        # Xóa các cột tạm
        if '__source_file__' in df_part.columns:
            del df_part['__source_file__']
        if '__original_index__' in df_part.columns:
            del df_part['__original_index__']

        output_name = f"{filename}_unique.csv"
        df_part.to_csv(output_name, index=False, encoding='utf-8-sig')
        print(f"📁 Đã lưu: {output_name} ({len(df_part)} dòng)")


# --- CHẠY ---
if __name__ == "__main__":
    if torch.cuda.is_available():
        process_gpu_duplicates()
    else:
        print("❌ Không tìm thấy GPU NVIDIA. Vui lòng cài đặt CUDA.")