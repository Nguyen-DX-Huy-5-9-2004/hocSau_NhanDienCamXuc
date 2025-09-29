import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import evaluate
import pickle
import time  

print("CUDA có sẵn:", torch.cuda.is_available())
print("Đang dùng thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
# 1. Load dữ liệu
df = pd.read_excel(r"D:\NguyenDucHuy\ZaloReceivedFiles\Fine-Grained_Emotion_Recognition_Dataset.xlsx", sheet_name='Sheet1')
df = df[["Câu", "Nhãn cảm xúc"]].dropna()

# 2. Encode nhãn
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Nhãn cảm xúc"])
num_labels = len(label_encoder.classes_)

# 3. Train/test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Câu"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# 4. Tải tokenizer và model PhoBERT
phobert_model = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(phobert_model, use_fast=False)

# 5. Tokenize dữ liệu
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize)
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels}).map(tokenize)

# 6. Load model với số lớp đúng
model = AutoModelForSequenceClassification.from_pretrained(phobert_model, num_labels=num_labels)

# 7. Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="phobert_emotion_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
)

# 9. Huấn luyện
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

start_time = time.time()
trainer.train()
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Tổng thời gian huấn luyện: {elapsed_time:.2f} giây")

# 10. Lưu mô hình và tokenizer
model.save_pretrained("phobert_emotion_model")
tokenizer.save_pretrained("phobert_emotion_model")

# Lưu encoder nhãn
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


