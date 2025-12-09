import os
# --- CẤU HÌNH LƯU CACHE SANG Ổ D ---
# Tạo thư mục này trên ổ D trước nếu chưa có
cache_dir = "D:/huggingface_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Thiết lập biến môi trường
os.environ["HF_HOME"] = cache_dir
import pandas as pd
import numpy as np
import torch
import re
import emoji
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- CẤU HÌNH ---
INPUT_FILE = "Data/step1/data_train_step1_balanced.csv"
OUTPUT_DIR = "./models/step1_mdeberta"  # Đổi tên folder output

# 🔥 THAY ĐỔI QUAN TRỌNG Ở ĐÂY 🔥
# Dùng mDeBERTa-v3-base: Kiến trúc mới hơn, mạnh hơn RoBERTa
MODEL_NAME = "microsoft/mdeberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def normalize_text(text):
    if not isinstance(text, str): return str(text) if text is not None else ""
    text = re.sub(r'[\/\\]\s*(\d*h|giờ|tiếng)', ' một giờ ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(ngày|day)', ' một ngày ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\/\\]\s*(tháng|month)', ' một tháng ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*(k|ka|xu)\b', r'\1 nghìn', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\d+)\s*(tr|triệu|củ)\b', r'\1 triệu', text, flags=re.IGNORECASE)
    return emoji.demojize(text, language='alias')


def preprocess_function(examples):
    # mDeBERTa xử lý tốt context dài, nhưng 512 là chuẩn
    return tokenizer(examples["text"], truncation=True, max_length=512)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def main():
    print(f"🚀 BẮT ĐẦU TRAIN MODEL: {MODEL_NAME}...")

    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {INPUT_FILE}")
        return

    df['text'] = df['text'].apply(normalize_text)

    # Đảm bảo label tồn tại
    if 'is_recruitment' not in df.columns:
        print("❌ File thiếu cột 'is_recruitment'")
        return

    df = df.dropna(subset=['is_recruitment'])
    df['label'] = df['is_recruitment'].astype(int)

    print(f"📊 Dữ liệu train: {df['label'].value_counts()}")

    # 2. Split Data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 3. Training Config
    training_args = TrainingArguments(
        output_dir="./results/filter_checkpoints_mdeberta",
        learning_rate=2e-5,  # mDeBERTa nên để learning rate nhỏ (1e-5 hoặc 2e-5)

        # RTX 3080 10GB có thể chịu được batch 8
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,

        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 4. Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Đã lưu model mDeBERTa tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()