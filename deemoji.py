import emoji
import pandas as pd

# 1. Đọc file
try:
    df = pd.read_csv("Data/facebook.csv")
    print("✅ Đã load file facebook.csv")
except FileNotFoundError:
    print("❌ Không tìm thấy file facebook.csv")


def convert_emoji(text):
    if not isinstance(text, str): # Nếu không phải chuỗi (ví dụ là nan/float)
        return str(text)
    return emoji.demojize(text, language='alias')


print("⏳ Đang chuyển đổi Emoji...")
df['title'] = df['tile'].apply(convert_emoji)
df['description'] = df['desc'].apply(convert_emoji)
df.__delitem__("tile")
df.__delitem__("desc")
# 5. Lưu file
output_file = "Data/data_demojized.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"💾 Đã lưu xong file: {output_file}")