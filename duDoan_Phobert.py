import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import torch.nn.functional as F

# 1. Load mô hình và tokenizer
model_dir = "phobert_emotion_model"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model.eval()

# 2. Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 3. In ra tất cả các nhãn có thể có
all_labels = list(label_encoder.classes_)
print("Các nhãn cảm xúc có thể dự đoán:")
for i, label in enumerate(all_labels):
    print(f"{i}: {label}")

# 4. Nhập câu và dự đoán
print("\nNhập câu để nhận diện cảm xúc (gõ 'exit' để thoát):")
while True:
    text = input("> ")
    if text.strip().lower() == "exit":
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        label = label_encoder.inverse_transform([predicted])[0]
        confidence = probs[0][predicted].item() * 100
        #them
        print("**Xác suất cho từng nhãn:")
    for idx, prob in enumerate(probs[0]):
        lbl = label_encoder.inverse_transform([idx])[0]
        print(f" - {lbl}: {prob.item() * 100:.2f}%")
    print(f"Cảm xúc dự đoán: {label} ({confidence:.2f}%)\n")
