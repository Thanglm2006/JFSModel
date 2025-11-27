import pandas as pd

df1 = pd.read_csv("data_demojized.csv")
df2 = pd.read_csv("data.csv")

df2["source"] = "facebook"
df_total = pd.concat([df1, df2], ignore_index=True)

df_shuffled = df_total.sample(frac=1, random_state=42).reset_index(drop=True)

df_shuffled.to_csv("data_train.csv", index=False, encoding='utf-8-sig')

print("✅ Đã gộp và trộn xong!")
print(df_shuffled.head())