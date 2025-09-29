import os
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re

DEFAULT_DATA_PATH = "D:/ZaloReceivedFiles/Fine-Grained_Emotion_Recognition_Dataset.xlsx"

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình BiLSTM nhận diện cảm xúc")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                        help="Đường dẫn đến file Excel chứa dữ liệu")
    parser.add_argument("--output_dir", type=str, default=".", help="Thư mục lưu checkpoint")
    parser.add_argument("--max_vocab", type=int, default=20000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()

def clean_text(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)  # loại dấu câu
    s = re.sub(r'\d+', '', s)      # loại số
    return s.strip()

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
        nn.init.xavier_uniform_(self.embedding.weight)
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

if __name__ == '__main__':
    import sys
    # Cho phép chạy bằng nút Run mà không cần dòng lệnh
    if len(sys.argv) == 1:  # không truyền đối số qua terminal
        class Args:
            data = "D:/ZaloReceivedFiles/Fine-Grained_Emotion_Recognition_Dataset.xlsx"
            output_dir = "./model_bilstm"
            max_vocab = 20000
            embed_dim = 128
            hidden_dim = 128
            num_layers = 2
            dropout = 0.3
            batch_size = 64
            lr = 1e-3
            weight_decay = 1e-5
            epochs = 30
        args = Args()
    else:
        args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang huấn luyện trên thiết bị: {device}")

    # 1. Đọc và xử lý dữ liệu
    df = pd.read_excel(args.data, sheet_name='Sheet1')
    df['text'] = df['Câu'].astype(str).apply(clean_text)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Nhãn cảm xúc'])

    print("\n📚 Danh sách nhãn:")
    for i, label in enumerate(le.classes_):
        print(f"{i}: {label}")

    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values, df['label'].values, test_size=0.2, stratify=df['label'], random_state=42)

    tokenized = [s.split() for s in X_train]
    freq = Counter(chain.from_iterable(tokenized))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in freq.most_common(args.max_vocab):
        vocab[word] = len(vocab)
    pad_idx = vocab['<PAD>']
    unk_idx = vocab['<UNK>']

    def encode_and_pad(sentences, max_len=None):
        encoded = [[vocab.get(w, unk_idx) for w in s.split()] for s in sentences]
        if max_len is None:
            max_len = max(len(s) for s in encoded)
        lengths = torch.tensor([min(len(s), max_len) for s in encoded])
        padded = [s[:max_len] + [pad_idx] * (max_len - len(s)) if len(s) < max_len else s[:max_len] for s in encoded]
        return torch.tensor(padded), lengths

    X_train_tensor, len_train = encode_and_pad(X_train)
    X_val_tensor, len_val = encode_and_pad(X_val, max_len=X_train_tensor.shape[1])
    y_train_tensor = torch.tensor(y_train)
    y_val_tensor = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor, len_train)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor, len_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = EmotionBiLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(le.classes_),
        pad_idx=pad_idx,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb, lb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad()
            out = model(xb, lb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                out = model(xb, lb)
                loss = criterion(out, yb)
                val_loss += loss.item() * yb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
                all_preds.extend(out.argmax(1).cpu().tolist())
                all_labels.extend(yb.cpu().tolist())

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"✅ Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_loss or val_acc > best_acc:
            best_loss = min(val_loss, best_loss)
            best_acc = max(val_acc, best_acc)
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'label_encoder': le,
                'max_len': X_train_tensor.shape[1],
                'pad_idx': pad_idx
            }, os.path.join(args.output_dir, "emotion_model_ckpt.pth"))
            print("💾 → Đã lưu mô hình tốt nhất.")

    print("\n📊 Classification Report trên tập validation:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
