# Chương trình gốc dùng TensorFlow/Keras để đọc bộ dữ liệu cảm xúc từ file Excel, tiền xử lý và huấn luyện một mạng neural LSTM 2 chiều. Các bước chính bao gồm: đọc dữ liệu (pd.read_excel), tiền xử lý văn bản (chuyển về chữ thường), mã hoá nhãn (LabelEncoder) và mã hoá từ (Keras Tokenizer + pad_sequences). Mô hình được xây dựng gồm lớp Embedding (128 chiều), theo sau là Bidirectional LSTM 128 đơn vị, thêm Dropout 0.3 và hai lớp Dense (64 neuron + softmax). Cuối cùng, mô hình được biên dịch với optimizer Adam và loss sparse-categorical-crossentropy, huấn luyện 100 epoch với batch size 64. Mô hình, tokenizer và label encoder được lưu lại sau khi huấn luyện. Những hạn chế của mô hình gốc bao gồm: chưa sử dụng embedding đã tiền huấn luyện (để tận dụng ngữ cảnh từ dữ liệu lớn), chưa có cơ chế attention hay cấu trúc CNN để trích xuất thêm đặc trưng từ câu, chỉ một lớp LSTM nên có thể thiếu khả năng học sâu hơn. Việc huấn luyện 100 epoch mà không có điều kiện dừng sớm (early stopping) có thể dẫn đến overfitting. Dữ liệu có thể chưa được cân bằng giữa các nhãn – mặc dù code có stratify trên nhãn khi chia train/test. Ngoài ra, chương trình gốc cũng kiểm tra và sử dụng GPU nếu có, tuy nhiên TensorFlow có phần khác biệt. PyTorch cũng hỗ trợ GPU – theo tài liệu, Tensor PyTorch có thể tận dụng GPU để tăng tốc hàng chục lần so với CPU (có thể ~50×)
# Do đó, việc sử dụng PyTorch trên GPU sẽ giúp huấn luyện nhanh hơn đáng kể.
# Chiến lược cải tiến và tối ưu mô hình
# Để tăng độ chính xác và hiệu quả của mô hình, ta có thể áp dụng một số cải tiến sau:
# Sử dụng embedding đã tiền huấn luyện (Word2Vec, GloVe, hoặc FastText) để khởi tạo lớp Embedding. Việc này giúp mô hình bắt đầu với kiến thức ngữ nghĩa phong phú từ dữ liệu lớn, thay vì học embedding từ đầu
# analyticsvidhya.com
# . Ví dụ, nghiên cứu của Analytics Vidhya khuyên dùng Bi-LSTM kết hợp Word2Vec để đạt kết quả tốt hơn trong nhận diện cảm xúc
# analyticsvidhya.com
# Thêm cơ chế Attention hoặc kết hợp CNN: Nhiều công trình cho thấy tích hợp cơ chế attention hoặc dùng thêm lớp CNN 1D trước/sau LSTM giúp trích xuất đặc trưng quan trọng và tăng độ chính xác. Một nghiên cứu trên tập dữ liệu IMDB đã chứng minh mô hình lai Bi-LSTM + CNN kèm attention cho kết quả phân loại chính xác hơn so với các mô hình LSTM hay CNN đơn lẻ
# mdpi.com
# . Chính vì vậy, ta có thể cân nhắc thêm lớp Conv1d hoặc cơ chế attention đơn giản để cải thiện.
# Áp dụng mô hình Transformer (BERT): Nếu yêu cầu độ chính xác càng cao càng tốt và dữ liệu cho phép, việc fine-tune một mô hình transformer hiện đại (như BERT hoặc RoBERTa) thường mang lại hiệu năng cao trong phân loại cảm xúc. Các nghiên cứu gần đây cho thấy BERT cải thiện đáng kể độ chính xác trong phân loại cảm xúc đa nhãn
# propulsiontechjournal.com
# . Đồng thời, tích hợp RNN/LSTM với BERT cũng có hiệu quả: một bài báo năm 2024 chỉ ra rằng mô hình kết hợp LSTM hai chiều và BERT đạt hiệu suất vượt trội nhờ LSTM bắt tốt thông tin thứ tự câu còn BERT cung cấp embedding ngữ cảnh phong phú
# mdpi.com
# . (Tuy nhiên trong mã minh họa dưới đây chúng ta sẽ ưu tiên kiến trúc Bi-LSTM kết hợp một số cải tiến ở trên.)
# Điều chỉnh siêu tham số và kỹ thuật huấn luyện: Dùng optimizer AdamW (Adam với weight decay) hoặc thêm weight decay để chống quá khớp. Giảm dần learning rate theo epoch hoặc dùng scheduler khi val loss không giảm. Áp dụng early stopping dựa trên validation loss để dừng huấn luyện khi không còn cải thiện. Điều chỉnh dropout (thêm một vài tầng dropout, tăng tỷ lệ nếu cần). Cân bằng dữ liệu bằng cách over-sample/under-sample nhãn thiểu số nếu phân bố nhãn mất cân bằng
# analyticsvidhya.com
import os
import time
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Thiết bị huấn luyện: {device}")

df = pd.read_excel(r"D:\NguyenDucHuy\ZaloReceivedFiles\Fine-Grained_Emotion_Recognition_Dataset.xlsx", sheet_name='Sheet1')
df['Câu'] = df['Câu'].astype(str).str.lower()

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['Nhãn cảm xúc'].tolist())
num_classes = len(label_encoder.classes_)

X_train_sent, X_test_sent, y_train, y_test = train_test_split(
    df['Câu'].values, labels, test_size=0.2, random_state=42, stratify=labels)

from collections import Counter
from itertools import chain

tokenized_train = [sentence.split() for sentence in X_train_sent]
counter = Counter(chain.from_iterable(tokenized_train))
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, _ in counter.most_common(20000):
    if word not in vocab:
        vocab[word] = len(vocab)
pad_idx = vocab['<PAD>']
unk_idx = vocab['<UNK>']

def encode_sentence(sent):
    return [vocab.get(tok, unk_idx) for tok in sent.split()]

encoded_train = [encode_sentence(sent) for sent in X_train_sent]
encoded_test  = [encode_sentence(sent) for sent in X_test_sent]
max_len = max(len(seq) for seq in encoded_train)
pickle.dump(max_len, open("max_len.pkl", "wb"))

def pad_sequences(sequences, max_len, pad_value=0):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [pad_value] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return torch.tensor(padded, dtype=torch.long)

X_train_tensor = pad_sequences(encoded_train, max_len, pad_idx)
X_test_tensor  = pad_sequences(encoded_test, max_len, pad_idx)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = self.attn(lstm_output).squeeze(-1)
        attn_weights = torch.softmax(weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
#bọn em ddax cai cải thiện bằng cách tăng số lớp của mô hình lên 4 và sử dụng attetion
class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super(EmotionBiLSTM, self).__init__()
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

model = EmotionBiLSTM(len(vocab), 128, 128, num_classes, pad_idx).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 100 #Bọn em đã tăg số bước phù hợp để tăng độ chính xác và tính tổng thể kết quả được tốt hơn và cân đối được thời gian huấn luyện, tranhs qua khớp
best_val_loss = float('inf')
manual_val_accs = []
manual_val_losses = []
start_manual = time.time()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = total_loss / total
    train_acc = correct / total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    manual_val_losses.append(val_loss)
    manual_val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_emotion_model.pt')
end_manual = time.time()
pickle.dump(manual_val_accs, open("manual_val_accs.pkl", "wb"))
pickle.dump(manual_val_losses, open("manual_val_losses.pkl", "wb"))

print(f"\nĐã huấn luyện thủ công xong. Thời gian: {end_manual - start_manual:.2f} giây")
class EmotionBiLSTM_NoAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super(EmotionBiLSTM_NoAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=4)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embeds)
        h = torch.cat((hn[-1], hn[-2]), dim=1) 
        x = self.dropout(h)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
# hl k atten, ddeer so sánh độ hiểu quả với dùng attention, vânx giữ nguyên số lớp
model_no_attn = EmotionBiLSTM_NoAttention(len(vocab), 128, 128, num_classes, pad_idx).to(device)
optimizer_no_attn = torch.optim.AdamW(model_no_attn.parameters(), lr=1e-3, weight_decay=1e-5)

no_attn_val_accs = []
no_attn_val_losses = []

start_no_attn = time.time()
for epoch in range(num_epochs):
    model_no_attn.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_no_attn.zero_grad()
        outputs = model_no_attn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_no_attn.step()
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = total_loss / total
    train_acc = correct / total

    model_no_attn.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_no_attn(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    no_attn_val_accs.append(val_acc)
    no_attn_val_losses.append(val_loss)

    print(f"(Không dùng atten - Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_no_attn.state_dict(), 'best_emotion_model_no_attn.pt')

end_no_attn = time.time()
pickle.dump(no_attn_val_accs, open("no_attn_val_accs.pkl", "wb"))
pickle.dump(no_attn_val_losses, open("no_attn_val_losses.pkl", "wb"))

print(f"\nĐã huấn luyện xong mô hình không dùng attention. Thời gian: {end_no_attn - start_no_attn:.2f} giây")

#mô hình thư viện pytorch lightning
import pytorch_lightning as pl

class EmotionDataset(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_test, y_test, batch_size=64):#ở đây, để giả biên độ giao động và tinhd ổn định của mô hình??? Biết là 1 gói lớn thì nó có ảnh hưởng làm tăng cái sự biến thiên lên lớn hơn nhưng mà cùng một bộ dữ liệu dù batch lớn hay nhỏ thì sau dần nó sẽ phải khớp dần và ổn định dần mới hợp lý chứ sao lại cứ dao động như thế, kể cả sau mỗi vòng data nó sẽ lấy lại gói data ngẫu nhiên nhưng mà vẫn trong bộ dữ liệu đấy mà, khó hiểu???
        super().__init__()
        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class EmotionClassifier(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.val_accs = []
        self.val_losses = []

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embeds)
        h = torch.cat((hn[0], hn[1]), dim=1)
        x = self.dropout(h)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.val_accs.append(acc.item())
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)

model_light = EmotionClassifier(len(vocab), 128, 128, num_classes, pad_idx)
data_module = EmotionDataset(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

trainer = pl.Trainer(
    max_epochs=20,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    logger=False
)
start_light = time.time()
trainer.fit(model_light, data_module)
end_light = time.time()

torch.save(model_light.state_dict(), 'emotion_model_lightning.pt')
pickle.dump(model_light.val_accs, open("lightning_val_accs.pkl", "wb"))
pickle.dump(model_light.val_losses, open("lightning_val_losses.pkl", "wb"))
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

np.save('label_encoder_classes.npy', label_encoder.classes_)
print("\nĐã huấn luyện bằng PyTorch Lightning")
print(f"Thời gian huấn luyện Lightning: {end_light - start_light:.2f} giây")

if all(os.path.exists(f) for f in ["manual_val_accs.pkl", "lightning_val_accs.pkl", "no_attn_val_accs.pkl"]):
    man_accs = pickle.load(open("manual_val_accs.pkl", "rb"))
    light_accs = pickle.load(open("lightning_val_accs.pkl", "rb"))
    no_attn_accs = pickle.load(open("no_attn_val_accs.pkl", "rb"))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(man_accs)+1), man_accs, label='Thủ công (có Attention)')
    plt.plot(range(1, len(no_attn_accs)+1), no_attn_accs, label='Thủ công (không Attention)')
    plt.plot(range(1, len(light_accs)+1), light_accs, label='thu viện')
    
    plt.xlabel("Epoch")
    plt.ylabel("độ chính xác")
    plt.legend()
    plt.grid(True)
    plt.show()
