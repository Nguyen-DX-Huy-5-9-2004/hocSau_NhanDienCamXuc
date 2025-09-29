import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import os
import warnings
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
torch.serialization.add_safe_globals([
    sklearn.preprocessing.LabelEncoder
])
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*")

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EmotionTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, pad_idx, dropout=0.1, ff_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        mask = (x == self.pad_idx)
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        attn_weights = self.attn_pool(x).squeeze(-1)
        attn_weights = torch.softmax(attn_weights.masked_fill(mask, -1e9), dim=1)
        x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return self.fc(x)

def clean_text(text):
    return str(text).lower()

def encode_input(text, vocab, pad_idx, unk_idx, max_len):
    tokens = clean_text(text).split()
    encoded = [vocab.get(w, unk_idx) for w in tokens]
    if len(encoded) < max_len:
        encoded += [pad_idx] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return torch.tensor([encoded])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="model_bilstm/emotion_transformer_ckpt.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Đang dùng thiết bị:", device)

    #checkpoint = torch.load(args.ckpt, map_location=device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)

    vocab = checkpoint["vocab"]
    pad_idx = checkpoint["pad_idx"]
    max_len = checkpoint["max_len"]
    label_encoder: LabelEncoder = checkpoint["label_encoder"]
    unk_idx = vocab.get("<UNK>", 1)

    model = EmotionTransformer(
        vocab_size=len(vocab),
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        num_classes=len(label_encoder.classes_),
        pad_idx=pad_idx,
        dropout=0.3
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    import torch.nn.functional as F 

while True:
    text = input("\nNhập câu cần dự đoán cảm xúc (hoặc 'exit' để thoát): ").strip()
    if text.lower() == "exit":
        break

    input_tensor = encode_input(text, vocab, pad_idx, unk_idx, max_len).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        emotion = label_encoder.inverse_transform([pred_idx])[0]

        print("Xác suất cho từng nhãn:")
        for idx, p in enumerate(prob[0].cpu()):
            lbl = label_encoder.inverse_transform([idx])[0]
            print(f" - {lbl}: {p.item() * 100:.2f}%")

        print(f"Cảm xúc dự đoán: {emotion} (Xác suất: {prob[0][pred_idx].item() * 100:.2f}%)")