import pandas as pd
import emoji
import os
from sklearn.utils import shuffle

# --- CẤU HÌNH ---
INPUT_FILES = [
    "Data/step2/content4_step2.csv",
    "Data/step2/content3_step2.csv",
    "Data/step2/content2_step2.csv",
    "Data/step2/content1_step2.csv",
    "Data/step2/facebook.csv",
    "Data/step2/data_viet.csv"
]

OUTPUT_TRAIN = "Data/step2/data_train_step2_balanced.csv"
TARGET_COL = 'label'  # Cột dùng để cân bằng


def convert_emoji(text):
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return emoji.demojize(text, language='alias')


def clean_and_merge():
    print("🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU CHO MODEL 1 (FILTER)...")

    all_dfs = []

    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            print(f"❌ Bỏ qua file không tồn tại: {file_path}")
            continue

        try:
            print(f"📂 Đang đọc: {file_path}")
            df = pd.read_csv(file_path)

            # 1. Chuẩn hóa tên cột TEXT
            if 'text' not in df.columns:
                if 'content' in df.columns:
                    df.rename(columns={'content': 'text'}, inplace=True)
                elif 'title' in df.columns and 'description' in df.columns:
                    df['text'] = df['title'].astype(str) + " " + df['description'].astype(str)
                else:
                    # Lấy cột đầu tiên làm text nếu không tìm thấy
                    df.rename(columns={df.columns[0]: 'text'}, inplace=True)

            # 2. Kiểm tra cột mục tiêu (is_recruitment)
            if TARGET_COL not in df.columns:
                print(f"   ⚠️ Cảnh báo: File này thiếu cột '{TARGET_COL}'. Sẽ bị bỏ qua bước cân bằng nếu không fix.")
                # Nếu bạn muốn gán mặc định (ví dụ file facebook.csv toàn là rác):
                # df[TARGET_COL] = 0

            # 3. Làm sạch Text
            df['text'] = df['text'].astype(str).str.strip()
            df = df[~df['text'].isin(['nan', ''])]  # Bỏ nan/rỗng
            df = df.dropna(subset=['text'])

            # 4. Chuyển đổi Emoji
            df['text'] = df['text'].apply(convert_emoji)

            # 5. Đánh dấu nguồn
            df['source_file'] = os.path.basename(file_path)

            all_dfs.append(df)

        except Exception as e:
            print(f"❌ Lỗi file {file_path}: {e}")

    # --- GỘP DỮ LIỆU ---
    if not all_dfs:
        return

    print("\n🔄 Đang gộp dữ liệu...")
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Xóa trùng lặp nội dung
    before = len(final_df)
    final_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    print(f"   - Đã loại bỏ {before - len(final_df)} dòng trùng lặp.")

    # --- CÂN BẰNG DỮ LIỆU DỰA TRÊN 'is_recruitment' ---
    print(f"\ngộp tỉ lệ theo cột '{TARGET_COL}'...")

    if TARGET_COL in final_df.columns:
        # Loại bỏ dòng mà is_recruitment bị null
        final_df = final_df.dropna(subset=[TARGET_COL])
        final_df[TARGET_COL] = final_df[TARGET_COL].astype(int)

        counts = final_df[TARGET_COL].value_counts()
        print("   - Phân bố gốc:", dict(counts))

        if len(counts) < 2:
            print("   ⚠️ Chỉ có 1 loại nhãn (toàn 0 hoặc toàn 1). Không thể cân bằng!")
        else:
            min_count = counts.min()
            # Lấy mẫu
            df_0 = final_df[final_df[TARGET_COL] == 0].sample(n=min_count, random_state=42)
            df_1 = final_df[final_df[TARGET_COL] == 1].sample(n=min_count*1, random_state=42)

            # Gộp và Trộn
            final_df = pd.concat([df_0, df_1])
            final_df = shuffle(final_df, random_state=42).reset_index(drop=True)

            print("   - Phân bố sau cân bằng:", dict(final_df[TARGET_COL].value_counts()))
    else:
        print(f"❌ Không tìm thấy cột '{TARGET_COL}' để cân bằng!")

    # --- LƯU FILE ---
    # Giữ nguyên tất cả các cột (text, is_recruitment, label, source_file...)
    final_df.to_csv(OUTPUT_TRAIN, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 40)
    print(f"🎉 HOÀN TẤT! File dùng để train Model 1: {OUTPUT_TRAIN}")
    print(f"📊 Tổng số dòng: {len(final_df)}")
    # In ra các cột để bạn kiểm tra xem cột 'label' còn đó không
    print(f"📋 Các cột hiện có: {list(final_df.columns)}")
    print("=" * 40)
    print(final_df[[TARGET_COL, 'label', 'text']].head())


if __name__ == "__main__":
    clean_and_merge()