import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from itertools import chain
import math
import time

import warnings
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="D:/NguyenDucHuy/ZaloReceivedFiles/Fine-Grained_Emotion_Recognition_Dataset.xlsx")
    parser.add_argument('--output_dir', type=str, default='./model_bilstm')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--max_vocab', type=int, default=20000)
    return parser.parse_args()

def build_vocab(texts, max_vocab):
    tokenized = [s.split() for s in texts]
    freq = Counter(chain.from_iterable(tokenized))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in freq.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab

def encode_and_pad(sentences, vocab, pad_idx, unk_idx, max_len=None):
    encoded = [[vocab.get(w, unk_idx) for w in s.split()] for s in sentences]
    if max_len is None:
        max_len = max(len(s) for s in encoded)
    lengths = torch.tensor([min(len(s), max_len) for s in encoded])
    padded = [s[:max_len] + [pad_idx]*(max_len - len(s)) if len(s) < max_len else s[:max_len] for s in encoded]
    return torch.tensor(padded), lengths

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"\nĐang huấn luyện trên thiết bị GPU: {gpu_name}")
    else:
        print(f"\nĐang huấn luyện trên thiết bị: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_excel(args.data, sheet_name='Sheet1')
    df['text'] = df['Câu'].astype(str).str.lower()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Nhãn cảm xúc'])

    X_train, X_val, y_train, y_val = train_test_split(df['text'].values, df['label'].values, test_size=0.2, stratify=df['label'], random_state=42)
    vocab = build_vocab(X_train, args.max_vocab)
    pad_idx = vocab['<PAD>']
    unk_idx = vocab['<UNK>']

    X_train_tensor, len_train = encode_and_pad(X_train, vocab, pad_idx, unk_idx)
    X_val_tensor, len_val = encode_and_pad(X_val, vocab, pad_idx, unk_idx, max_len=X_train_tensor.shape[1])
    y_train_tensor = torch.tensor(y_train)
    y_val_tensor = torch.tensor(y_val)

    train_ds = TensorDataset(X_train_tensor, y_train_tensor, len_train)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor, len_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = EmotionTransformer(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=len(le.classes_),
        pad_idx=pad_idx,
        dropout=args.dropout,
        ff_dim=args.ff_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb, lb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (output.argmax(1) == yb).sum().item()
            total += yb.size(0)

        val_correct, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                val_correct += (output.argmax(1) == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: train_loss={total_loss/total:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'label_encoder': le,
                'pad_idx': pad_idx,
                'max_len': X_train_tensor.shape[1]
            }, os.path.join(args.output_dir, "emotion_transformer_ckpt.pth"))
            print("Đã lưu mô hình tốt nhất!")
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nThời gian huấn luyện tổng cộng: {elapsed_minutes:.2f} phút")
    print("\nĐang đánh giá mô hình tốt nhất trên tập validation...")
    checkpoint = torch.load(os.path.join(args.output_dir, "emotion_transformer_ckpt.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb, lb in val_loader:
            xb = xb.to(device)
            output = model(xb)
            preds = output.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(yb.tolist())

    from sklearn.metrics import classification_report, accuracy_score

    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    print(f"Độ chính xác (Validation Accuracy): {accuracy_score(all_labels, all_preds):.4f}")