import re

import emoji
import pandas as pd
import numpy as np
import torch
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def preprocess_function(examples):
    # Lưu ý: biến tokenizer phải được truyền vào hoặc khai báo global
    return tokenizer(examples["text"], truncation=True, max_length=512)
def normalize_text(text):
    # 1. Kiểm tra an toàn trước tiên
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # LƯU Ý: Đã xóa dòng text = text.lower()

    # 2. Chuẩn hóa thời gian/lương (Thêm flags=re.IGNORECASE)
    # Bắt: /h, /H, /giờ, /Giờ...
    text = re.sub(r'[\/\\]\s*(\d*h|giờ|tiếng)', ' một giờ ', text, flags=re.IGNORECASE)

    # Bắt: /ngày, /Day, /Ngày...
    text = re.sub(r'[\/\\]\s*(ngày|day)', ' một ngày ', text, flags=re.IGNORECASE)

    # Bắt: /tháng, /Month...
    text = re.sub(r'[\/\\]\s*(tháng|month)', ' một tháng ', text, flags=re.IGNORECASE)

    # 3. Chuẩn hóa đơn vị tiền
    # Bắt: 100k, 100K, 100ka, 100KA
    text = re.sub(r'\b(\d+)\s*(k|ka|xu)\b', r'\1 nghìn', text, flags=re.IGNORECASE)

    # Bắt: 5tr, 5TR, 5Tr, 5củ...
    text = re.sub(r'\b(\d+)\s*(tr|triệu|củ)\b', r'\1 triệu', text, flags=re.IGNORECASE)

    # 4. Demojize (Chuyển icon thành text :smile:)
    return emoji.demojize(text, language='alias')

# --- QUAN TRỌNG: TOÀN BỘ CODE CHẠY PHẢI NẰM TRONG KHỐI NÀY ---
if __name__ == "__main__":

    # 1. KIỂM TRA GPU
    print("-" * 30)
    if torch.cuda.is_available():
        print(f"✅ Đã tìm thấy GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Chạy trên CPU")
    print("-" * 30)

    # 2. LOAD DATA
    df = pd.read_csv("data_train.csv")
    print("loaded data!")

    if 'text' not in df.columns:
        df['text'] = normalize_text(df['title'].astype(str) + " \n " + df['description'].astype(str))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 3. LOAD MODEL & TOKENIZER
    model_name = "xlm-roberta-base"
    # Khai báo tokenizer global để hàm preprocess dùng được
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. CONFIG TRAINING
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        save_total_limit=2,
        learning_rate=2e-5,

        # Cấu hình cho RTX 3080 10GB
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        fp16=True,

        # --- QUAN TRỌNG CHO WINDOWS: PHẢI ĐỂ LÀ 0 ---
        dataloader_num_workers=0,
        # --------------------------------------------

        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
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

    print("🚀 Bắt đầu training!")
    trainer.train()

    # 5. SAVE MODEL
    save_path = "./my_scam_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"💾 Đã lưu model tại: {save_path}")