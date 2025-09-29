import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
from torch.nn.utils.rnn import pad_sequence

# ---------- Hàm xử lý văn bản ----------
def clean_text(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\d+', '', s)
    return s.strip()

# ---------- Attention và Model ----------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out, mask=None):
        scores = self.attn(lstm_out).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context

class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        embeds = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        max_len_out = lstm_out.size(1)
        mask = torch.arange(max_len_out, device=lengths.device)[None, :] < lengths[:, None]
        context = self.attention(lstm_out, mask)
        return self.fc(context)

# ---------- Load mô hình ----------
ckpt_path = "./model_bilstm/emotion_model_ckpt.pth"
#checkpoint = torch.load(ckpt_path, map_location="cpu")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
vocab = checkpoint['vocab']
label_encoder = checkpoint['label_encoder']
max_len = checkpoint['max_len']
pad_idx = checkpoint['pad_idx']

model = EmotionBiLSTM(
    vocab_size=len(vocab),
    embed_dim=128,
    hidden_dim=128,
    num_classes=len(label_encoder.classes_),
    pad_idx=pad_idx,
    num_layers=2,
    dropout=0.3
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------- Hàm mã hóa câu ----------
def encode_sentence(sentence, vocab, max_len, pad_idx, unk_idx=1):
    tokens = clean_text(sentence).split()
    ids = [vocab.get(tok, unk_idx) for tok in tokens]
    if len(ids) < max_len:
        ids += [pad_idx] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    length = torch.tensor([min(len(tokens), max_len)])
    return torch.tensor([ids]), length

# ---------- Dự đoán ----------
print("\n💬 Nhập câu để nhận diện cảm xúc (gõ 'exit' để thoát):")
while True:
    text = input("> ")
    if text.strip().lower() == "exit":
        break

    x_tensor, length_tensor = encode_sentence(text, vocab, max_len, pad_idx)
    with torch.no_grad():
        outputs = model(x_tensor, length_tensor)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = label_encoder.inverse_transform([pred])[0]
        confidence = probs[0][pred].item() * 100

    print(f"🔎 Cảm xúc dự đoán: {label} ({confidence:.2f}%)\n")
