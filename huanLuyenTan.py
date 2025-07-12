# --------------------------
# File: train_emotion_model.py (GPU Optimized Version)
# --------------------------

import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import pickle

# --------------------------
# Kiểm tra và thiết lập GPU
# --------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✔️  Đã tìm thấy {len(gpus)} GPU. Sử dụng GPU để train.")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️  Không tìm thấy GPU, sẽ train trên CPU.")

# --------------------------
# Đọc dữ liệu Excel
# --------------------------
df = pd.read_excel(r'D:/Python/Fine-Grained Emotion Recognition Dataset (1).xlsx', sheet_name='Sheet1')

# --------------------------
# Phân tích phân bố nhãn
# --------------------------
print("Phân bố các nhãn:")
print(df['Nhãn cảm xúc'].value_counts())

# --------------------------
# Tiền xử lý câu
# --------------------------
def clean_text(text):
    return text.lower()

sentences = df['Câu'].astype(str).apply(clean_text).tolist()
emotions = df['Nhãn cảm xúc'].tolist()

# --------------------------
# Encode nhãn cảm xúc
# --------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(emotions)

# --------------------------
# Tokenize câu
# --------------------------
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
max_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# --------------------------
# Chia train/test
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# Tạo mô hình Bidirectional LSTM
# --------------------------
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# Train mô hình lâu hơn với GPU hỗ trợ
# --------------------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=2
)

# --------------------------
# Lưu model và tokenizer
# --------------------------
model.save('emotion_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\nĐã train xong và lưu model. Max length của câu: {max_length}")


