# Dự đoán cảm xúc từ câu nói sử dụng 2 mô hình: THỦ CÔNG và PYTORCH LIGHTNING

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# --------------------------
# Load dữ liệu: vocab, label encoder, max_len
# --------------------------
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)

pad_idx = vocab['<PAD>']
unk_idx = vocab['<UNK>']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Hàm encode và pad câu nhập
# --------------------------
def encode_sentence(sent, vocab, unk_idx):
    return [vocab.get(tok, unk_idx) for tok in sent.lower().split()]

def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) < max_len:
        return seq + [pad_value] * (max_len - len(seq))
    else:
        return seq[:max_len]

# --------------------------
# Mô hình THỦ CÔNG (BiLSTM + Attention + 4 lớp LSTM)
# --------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = self.attn(lstm_output).squeeze(-1)
        attn_weights = torch.softmax(weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=4)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        context = self.attention(lstm_out)
        x = self.dropout(context)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --------------------------
# Mô hình LIGHTNING (1 lớp LSTM, không có attention)
# --------------------------
class EmotionClassifier(nn.Module):  # Không cần LightningModule khi dùng dự đoán
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embeds)
        h = torch.cat((hn[0], hn[1]), dim=1)
        x = self.dropout(h)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --------------------------
# Hàm dự đoán với mô hình bất kỳ
# --------------------------
def predict_emotion(model, sentence, max_len, vocab, label_encoder, device):
    model.eval()
    encoded = encode_sentence(sentence, vocab, unk_idx)
    padded = pad_sequence(encoded, max_len, pad_value=pad_idx)
    input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# --------------------------
# Load mô hình THỦ CÔNG
# --------------------------
print("\n--- Dự đoán bằng mô hình THỦ CÔNG (BiLSTM + Attention + 4 lớp) ---")
manual_model = EmotionBiLSTM(len(vocab), 128, 128, len(label_encoder.classes_), pad_idx).to(device)
manual_model.load_state_dict(torch.load("best_emotion_model.pt", map_location=device))

# --------------------------
# Load mô hình PYTORCH LIGHTNING
# --------------------------
print("\n--- Dự đoán bằng mô hình PYTORCH LIGHTNING ---")
lightning_model = EmotionClassifier(len(vocab), 128, 128, len(label_encoder.classes_), pad_idx).to(device)
lightning_model.load_state_dict(torch.load("emotion_model_lightning.pt", map_location=device))

# --------------------------
# Giao diện nhập liệu
# --------------------------
while True:
    sentence = input("\nNhập câu cần nhận diện cảm xúc (hoặc gõ 'exit' để thoát): ").strip()
    if sentence.lower() == 'exit':
        break

    pred_manual = predict_emotion(manual_model, sentence, max_len, vocab, label_encoder, device)
    pred_light = predict_emotion(lightning_model, sentence, max_len, vocab, label_encoder, device)

    print(f"\nMô hình thủ công dự đoán: {pred_manual}")
    print(f"Mô hình Lightning dự đoán: {pred_light}")