# --------------------------
# File: dudoan_cx.py (Optimized)
# --------------------------

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------
# Load model, tokenizer, label encoder
# --------------------------
model = load_model('emotion_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# --------------------------
# Tham số padding (auto load từ file train)
# --------------------------
# Nếu bạn lưu max_length vào 1 file txt thì đọc lại, ở đây giả sử là 40
max_length = 40

# --------------------------
# Hàm xử lý đầu vào giống khi train
# --------------------------
def clean_text(text):
    return text.lower()

# --------------------------
# Chạy nhận input và dự đoán
# --------------------------
print("=== EMOTION DETECTOR ===")
print("Gõ 'exit' để thoát")

while True:
    sentence = input("Nhập câu của bạn: ").strip()
    if sentence.lower() == 'exit':
        break

    cleaned_sentence = clean_text(sentence)

    seq = tokenizer.texts_to_sequences([cleaned_sentence])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    pred = model.predict(padded, verbose=0)
    emotion = label_encoder.inverse_transform([np.argmax(pred)])

    print(f"\u27a1\ufe0f  Cảm xúc được nhận diện: {emotion[0]}\n")

print("\n\ud83d\udce4 Đã thoát chương trình.")