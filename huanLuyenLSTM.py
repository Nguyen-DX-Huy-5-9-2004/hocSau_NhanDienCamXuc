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
# Tận dụng GPU với PyTorch: Đảm bảo gửi mô hình và dữ liệu lên thiết bị cuda (nếu có). Ví dụ: device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model.to(device)... Việc này theo tài liệu giúp tăng tốc huấn luyện đáng kể
# Định nghĩa và huấn luyện mô hình PyTorch
# Dưới đây là ví dụ mã Python dùng PyTorch để xây dựng, huấn luyện mô hình Bi-LSTM trên GPU (nếu có) với các cải tiến kể trên. Chú ý rằng ta sử dụng torch.utils.data.DataLoader để xử lý batch và GPU, đồng thời có thể dễ dàng mở rộng thêm lớp attention/CNN nếu cần.
# #DÙNG MÔ HÌNH THỦ CÔNG
# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader

# # --------------------------
# # Thiết lập thiết bị (GPU/CPU)
# # --------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Thiết bị huấn luyện: {device}")

# # --------------------------
# # Đọc và tiền xử lý dữ liệu
# # --------------------------
# df = pd.read_excel(r"D:\ZaloReceivedFiles\Fine-Grained_Emotion_Recognition_Dataset.xlsx", sheet_name='Sheet1')
# df['Câu'] = df['Câu'].astype(str).str.lower()  # đơn giản hóa: chỉ chuyển về chữ thường
# # Có thể thêm loại bỏ dấu câu, stopwords nếu cần

# # Mã hóa nhãn
# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(df['Nhãn cảm xúc'].tolist())
# num_classes = len(label_encoder.classes_)

# # Tách tập train/test (stratify để cân bằng nhãn)
# X_train_sent, X_test_sent, y_train, y_test = train_test_split(
#     df['Câu'].values, labels, test_size=0.2, random_state=42, stratify=labels)

# # --------------------------
# # Xây dựng từ điển (vocabulary) và mã hóa câu thành chuỗi số
# # --------------------------
# from collections import Counter
# from itertools import chain

# # Cách đơn giản: tách từ bằng dấu cách (có thể thay bằng tokenizer nâng cao)
# tokenized_train = [sentence.split() for sentence in X_train_sent]
# counter = Counter(chain.from_iterable(tokenized_train))
# # Giới hạn vocab tối đa (ví dụ 20000 từ), hoặc giữ hết
# vocab = {'<PAD>': 0, '<UNK>': 1}
# for word, _ in counter.most_common(20000):
#     if word not in vocab:
#         vocab[word] = len(vocab)
# pad_idx = vocab['<PAD>']
# unk_idx = vocab['<UNK>']

# def encode_sentence(sent):
#     tokens = sent.split()
#     return [vocab.get(tok, unk_idx) for tok in tokens]

# # Mã hóa và đệm (pad) thủ công
# encoded_train = [encode_sentence(sent) for sent in X_train_sent]
# encoded_test  = [encode_sentence(sent) for sent in X_test_sent]
# max_len = max(len(seq) for seq in encoded_train)  # chiều dài tối đa
# # Tạo tensor và điền pad
# def pad_sequences(sequences, max_len, pad_value=0):
#     padded = []
#     for seq in sequences:
#         if len(seq) < max_len:
#             padded.append(seq + [pad_value] * (max_len - len(seq)))
#         else:
#             padded.append(seq[:max_len])
#     return torch.tensor(padded, dtype=torch.long)

# X_train_tensor = pad_sequences(encoded_train, max_len, pad_value=pad_idx)
# X_test_tensor  = pad_sequences(encoded_test,  max_len, pad_value=pad_idx)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# y_test_tensor  = torch.tensor(y_test,  dtype=torch.long)

# # --------------------------
# # Đóng gói DataLoader
# # --------------------------
# batch_size = 64
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# # --------------------------
# # Định nghĩa mô hình Bi-LSTM
# # --------------------------
# class EmotionBiLSTM(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
#         super(EmotionBiLSTM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
#         # Có thể khởi tạo với embedding đã tiền huấn luyện ở đây nếu có
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(hidden_dim * 2, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         # x: [batch_size, seq_len]
#         embeds = self.embedding(x)  # [batch, seq_len, embed_dim]
#         lstm_out, (hn, cn) = self.lstm(embeds)
#         # Lấy trạng thái ẩn cuối (forward và backward) của LSTM 2 chiều
#         # hn: [num_layers*2, batch, hidden_dim] (với num_layers=1)
#         h_forward = hn[0]  # đầu ra cuối của chiều tiến
#         h_backward = hn[1] # đầu ra cuối của chiều ngược
#         h = torch.cat((h_forward, h_backward), dim=1)  # [batch, 2*hidden_dim]
#         x = self.dropout(h)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)  # logits cho mỗi lớp
#         return x

# vocab_size = len(vocab)
# embed_dim = 128
# hidden_dim = 128
# model = EmotionBiLSTM(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx)
# model.to(device)

# # --------------------------
# # Thiết lập hàm mất mát và optimizer
# # --------------------------
# criterion = nn.CrossEntropyLoss()  # tương đương sparse_categorical_crossentropy
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# # --------------------------
# # Vòng lặp huấn luyện
# # --------------------------
# num_epochs = 20
# best_val_loss = float('inf')
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     correct = 0
#     total = 0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)  # [batch, num_classes]
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * inputs.size(0)
#         preds = torch.argmax(outputs, dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#     train_loss = total_loss / total
#     train_acc = correct / total

#     # Đánh giá trên tập kiểm tra
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * inputs.size(0)
#             preds = torch.argmax(outputs, dim=1)
#             val_correct += (preds == labels).sum().item()
#             val_total += labels.size(0)
#     val_loss /= val_total
#     val_acc = val_correct / val_total

#     print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, " +
#           f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
#     # Early stopping đơn giản
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'best_emotion_model.pt')  # lưu model tốt nhất
    # Còn có thể thêm: giảm lr nếu val_loss không giảm

# --------------------------
# Lưu tokenizer và label encoder (bằng pickle hoặc tự định dạng)
# --------------------------
# import pickle
# with open('vocab.pkl', 'wb') as f:
#     pickle.dump(vocab, f)
# with open('label_encoder.pkl', 'wb') as f:
#     pickle.dump(label_encoder, f)

# print("Đã huấn luyện xong và lưu mô hình cùng dữ liệu cần thiết.")



# Trong mã trên, chúng ta đã:
# Sử dụng PyTorch DataLoader để lặp qua dữ liệu trên GPU.
# Dùng lớp Embedding tích hợp vào EmotionBiLSTM, với padding index để lớp embedding bỏ qua từ đệm. (Nếu có thể, ta có thể thay thế bằng embedding GloVe tiền huấn luyện cho vocab hiện tại.)
# Định nghĩa nn.LSTM hai chiều (bidirectional) 128 đơn vị, giống về cấu trúc với Keras ban đầu.
# Kết hợp hai chiều bằng cách ghép (concatenate) hai trạng thái ẩn cuối.
# Thêm Dropout và hai tầng Linear để phân loại.
# Dùng AdamW (Adam với weight decay) để cải thiện khả năng tổng quát.
# Huấn luyện với vòng lặp thủ công, tính loss và accuracy mỗi epoch, có lưu mô hình tốt nhất dựa trên val_loss.
# Các cải tiến có thể thêm: chèn CNN 1D trước LSTM để trích xuất đặc trưng cấp cao, hoặc thêm một đầu Attention sau LSTM để tập trung vào từ quan trọng. Ngoài ra, nếu dữ liệu lớn, có thể thử fine-tune BERT/PyTorch Transformer cho độ chính xác tối ưu (theo các nghiên cứu gần đây, mô hình lai BERT+RNN cải thiện rõ độ chính xác phân loại
# mdpi.com
# Kết quả: Với cách tiếp cận này, mô hình PyTorch sẽ huấn luyện nhanh hơn nhờ GPU và các cải tiến nêu trên. Nghiên cứu cho thấy việc dùng mô hình lai (như kết hợp Bi-LSTM, CNN và attention) cho kết quả phân loại chính xác hơn đáng kể
# . Nếu điều kiện cho phép, fine-tune một mô hình transformer như BERT có thể tiếp tục nâng cao độ chính xác tối đa cho bài toán nhận diện cảm xúc
# . Tóm lại, việc chuyển sang PyTorch và tối ưu mô hình bao gồm: sử dụng embedding đã huấn luyện, mạng LSTM hai chiều với dropout, thêm các thành phần như CNN/Attention, fine-tune siêu tham số, và tận dụng GPU để tăng tốc. Những cải tiến này sẽ giúp mô hình thu được độ chính xác huấn luyện cao nhất có thể trên tập dữ liệu cho trước. Nguồn tham khảo: Các phương pháp và ý tưởng trên được tham khảo từ các nghiên cứu và bài viết về phân loại văn bản – ví dụ, kết hợp Bi-LSTM với Word2Vec cho hiệu quả cao hơn
# , mô hình lai Bi-LSTM+CNN+Attention cải thiện độ chính xác phân loại
# , cũng như xu hướng dùng mô hình Transformer (BERT) trong phân loại cảm xúc đa nhãn
# . PyTorch cho phép huấn luyện trên GPU với tốc độ tăng đáng kể
# , giúp quá trình đào tạo hiệu quả hơn.





# #DÙNG MÔ HÌNH THỦ CÔNG (CÓ ATTENTION VÀ 4 LỚP LSTM)
# import os
# import time
# import pickle
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Thiết bị huấn luyện: {device}")

# df = pd.read_excel(r"D:\ZaloReceivedFiles\Fine-Grained_Emotion_Recognition_Dataset.xlsx", sheet_name='Sheet1')
# df['Câu'] = df['Câu'].astype(str).str.lower()

# label_encoder = LabelEncoder()
# labels = label_encoder.fit_transform(df['Nhãn cảm xúc'].tolist())
# num_classes = len(label_encoder.classes_)

# X_train_sent, X_test_sent, y_train, y_test = train_test_split(
#     df['Câu'].values, labels, test_size=0.2, random_state=42, stratify=labels)

# from collections import Counter
# from itertools import chain

# tokenized_train = [sentence.split() for sentence in X_train_sent]
# counter = Counter(chain.from_iterable(tokenized_train))
# vocab = {'<PAD>': 0, '<UNK>': 1}
# for word, _ in counter.most_common(20000):
#     if word not in vocab:
#         vocab[word] = len(vocab)
# pad_idx = vocab['<PAD>']
# unk_idx = vocab['<UNK>']

# def encode_sentence(sent):
#     return [vocab.get(tok, unk_idx) for tok in sent.split()]

# encoded_train = [encode_sentence(sent) for sent in X_train_sent]
# encoded_test  = [encode_sentence(sent) for sent in X_test_sent]
# max_len = max(len(seq) for seq in encoded_train)
# pickle.dump(max_len, open("max_len.pkl", "wb"))

# def pad_sequences(sequences, max_len, pad_value=0):
#     padded = []
#     for seq in sequences:
#         if len(seq) < max_len:
#             padded.append(seq + [pad_value] * (max_len - len(seq)))
#         else:
#             padded.append(seq[:max_len])
#     return torch.tensor(padded, dtype=torch.long)

# X_train_tensor = pad_sequences(encoded_train, max_len, pad_idx)
# X_test_tensor  = pad_sequences(encoded_test, max_len, pad_idx)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

# batch_size = 64
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = nn.Linear(hidden_dim * 2, 1)

#     def forward(self, lstm_output):
#         weights = self.attn(lstm_output).squeeze(-1)
#         attn_weights = torch.softmax(weights, dim=1)
#         context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
#         return context

# class EmotionBiLSTM(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
#         super(EmotionBiLSTM, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=4)
#         self.attention = Attention(hidden_dim)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(hidden_dim * 2, 64)
#         self.fc2 = nn.Linear(64, num_classes)

#     def forward(self, x):
#         embeds = self.embedding(x)
#         lstm_out, _ = self.lstm(embeds)
#         context = self.attention(lstm_out)
#         x = self.dropout(context)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

# model = EmotionBiLSTM(len(vocab), 128, 128, num_classes, pad_idx).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# num_epochs = 100
# best_val_loss = float('inf')
# manual_val_accs = []
# manual_val_losses = []
# start_manual = time.time()
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * inputs.size(0)
#         preds = outputs.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#     train_loss = total_loss / total
#     train_acc = correct / total

#     model.eval()
#     val_loss = 0
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * inputs.size(0)
#             preds = outputs.argmax(dim=1)
#             val_correct += (preds == labels).sum().item()
#             val_total += labels.size(0)
#     val_loss /= val_total
#     val_acc = val_correct / val_total

#     manual_val_losses.append(val_loss)
#     manual_val_accs.append(val_acc)

#     print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), 'best_emotion_model.pt')
# end_manual = time.time()
# pickle.dump(manual_val_accs, open("manual_val_accs.pkl", "wb"))
# pickle.dump(manual_val_losses, open("manual_val_losses.pkl", "wb"))

# print(f"\nĐã huấn luyện thủ công (có attention và 4 lớp LSTM) xong trong {end_manual - start_manual:.2f} giây")

# import pytorch_lightning as pl

# class EmotionDataset(pl.LightningDataModule):
#     def __init__(self, X_train, y_train, X_test, y_test, batch_size=64):
#         super().__init__()
#         self.train_dataset = TensorDataset(X_train, y_train)
#         self.test_dataset = TensorDataset(X_test, y_test)
#         self.batch_size = batch_size

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size)

# class EmotionClassifier(pl.LightningModule):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1 = nn.Linear(hidden_dim * 2, 64)
#         self.fc2 = nn.Linear(64, num_classes)
#         self.criterion = nn.CrossEntropyLoss()
#         self.val_accs = []
#         self.val_losses = []

#     def forward(self, x):
#         embeds = self.embedding(x)
#         lstm_out, (hn, _) = self.lstm(embeds)
#         h = torch.cat((hn[0], hn[1]), dim=1)
#         x = self.dropout(h)
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         acc = (logits.argmax(dim=1) == y).float().mean()
#         self.log("train_loss", loss, prog_bar=True)
#         self.log("train_acc", acc, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = self.criterion(logits, y)
#         acc = (logits.argmax(dim=1) == y).float().mean()
#         self.val_accs.append(acc.item())
#         self.val_losses.append(loss.item())
#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_acc", acc, prog_bar=True)

#     def configure_optimizers(self):
#         return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)

# model_light = EmotionClassifier(len(vocab), 128, 128, num_classes, pad_idx)
# data_module = EmotionDataset(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# trainer = pl.Trainer(
#     max_epochs=20,
#     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#     devices=1,
#     logger=False
# )
# start_light = time.time()
# trainer.fit(model_light, data_module)
# end_light = time.time()

# torch.save(model_light.state_dict(), 'emotion_model_lightning.pt')
# pickle.dump(model_light.val_accs, open("lightning_val_accs.pkl", "wb"))
# pickle.dump(model_light.val_losses, open("lightning_val_losses.pkl", "wb"))

# print("\nĐã huấn luyện bằng PyTorch Lightning. Model đã lưu là 'emotion_model_lightning.pt'")
# print(f"Thời gian huấn luyện Lightning: {end_light - start_light:.2f} giây")

# if os.path.exists("manual_val_accs.pkl") and os.path.exists("lightning_val_accs.pkl"):
#     man_accs = pickle.load(open("manual_val_accs.pkl", "rb"))
#     light_accs = pickle.load(open("lightning_val_accs.pkl", "rb"))
#     plt.plot(range(1, len(man_accs)+1), man_accs, label='Thủ công - val_acc')
#     plt.plot(range(1, len(light_accs)+1), light_accs, label='Lightning - val_acc')
#     plt.title("So sánh độ chính xác validation giữa hai mô hình")
#     plt.xlabel("Epoch")
#     plt.ylabel("Validation Accuracy")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


import os
import time
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

DEFAULT_DATA_PATH = "D:/ZaloReceivedFiles/Fine-Grained_Emotion_Recognition_Dataset.xlsx"

def parse_args():
    # Thiết lập các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện cảm xúc")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH,
                       help="Đường dẫn đến file dữ liệu")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Thư mục lưu kết quả")
    parser.add_argument("--max_vocab", type=int, default=20000, 
                       help="Kích thước từ điển tối đa")
    parser.add_argument("--embed_dim", type=int, default=128, 
                       help="Chiều embedding vector")
    parser.add_argument("--hidden_dim", type=int, default=128, 
                       help="Kích thước lớp ẩn LSTM")
    parser.add_argument("--num_layers", type=int, default=4, 
                       help="Số lớp LSTM (đã tăng lên 4)")
    parser.add_argument("--dropout", type=float, default=0.3, 
                       help="Tỉ lệ dropout")
    parser.add_argument("--batch_size", type=int, default=64, 
                       help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Tốc độ học")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                       help="Hệ số giảm trọng số")
    parser.add_argument("--epochs", type=int, default=30, 
                       help="Số epoch huấn luyện")
    parser.add_argument("--pretrained_model", type=str, default="vinai/phobert-base",
                       help="Mô hình tiền huấn luyện")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Độ dài tối đa của chuỗi đầu vào")
    return parser.parse_args()

class EmotionDataset(torch.utils.data.Dataset):
    # Lớp Dataset cho Transformers
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class Attention(nn.Module):
    # Lớp Attention cho mô hình LSTM
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
    # Mô hình BiLSTM với Attention tự xây dựng
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
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

def train_transformers_model(train_texts, train_labels, val_texts, val_labels, args, le):
    # Huấn luyện mô hình sử dụng thư viện Transformers
    print("\n=== Khởi tạo mô hình Transformers ===")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model, 
        num_labels=len(le.classes_)
    )

    # Tokenize dữ liệu với padding và truncation
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=args.max_length)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=args.max_length)

    # Tạo dataset
    train_dataset = EmotionDataset(train_encodings, train_labels)
    val_dataset = EmotionDataset(val_encodings, val_labels)

    # Thiết lập tham số huấn luyện
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "transformers"),
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Huấn luyện và đánh giá
    print("\n=== Bắt đầu huấn luyện Transformers ===")
    trainer.train()
    results = trainer.evaluate()
    
    return model, results

def train_custom_model(X_train_tensor, len_train, y_train_tensor, X_val_tensor, len_val, y_val_tensor, 
                      vocab, le, args, device):
    # Huấn luyện mô hình LSTM tự xây dựng
    print("\n=== Khởi tạo mô hình LSTM tự xây dựng ===")
    
    # Tạo DataLoader
    train_ds = TensorDataset(X_train_tensor, y_train_tensor, len_train)
    val_ds = TensorDataset(X_val_tensor, y_val_tensor, len_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Khởi tạo mô hình
    model = EmotionBiLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(le.classes_),
        pad_idx=vocab['<PAD>'],
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # Thiết lập optimizer và loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # Vòng lặp huấn luyện
    print("\n=== Bắt đầu huấn luyện LSTM tự xây dựng ===")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        
        # Huấn luyện trên từng batch
        for xb, yb, lb in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            
            optimizer.zero_grad()
            out = model(xb, lb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * yb.size(0)
            total_correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        
        # Tính toán metrics
        train_loss = total_loss / total
        train_acc = total_correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Đánh giá trên validation set
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                out = model(xb, lb)
                loss = criterion(out, yb)
                
                val_loss += loss.item() * yb.size(0)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Lưu mô hình tốt nhất
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'label_encoder': le,
                'max_len': X_train_tensor.shape[1],
                'pad_idx': vocab['<PAD>']
            }, os.path.join(args.output_dir, "custom_model_ckpt.pth"))
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def plot_comparison(transformers_results, custom_results, args):
    # Vẽ biểu đồ so sánh kết quả
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ loss
    plt.subplot(1, 2, 1)
    plt.plot(custom_results['train_losses'], label='LSTM Train Loss')
    plt.plot(custom_results['val_losses'], label='LSTM Val Loss')
    plt.axhline(y=transformers_results['eval_loss'], color='r', linestyle='--', 
                label=f'Transformers Val Loss: {transformers_results["eval_loss"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    
    # Biểu đồ accuracy
    plt.subplot(1, 2, 2)
    plt.plot(custom_results['train_accs'], label='LSTM Train Acc')
    plt.plot(custom_results['val_accs'], label='LSTM Val Acc')
    plt.axhline(y=transformers_results['eval_accuracy'], color='r', linestyle='--', 
                label=f'Transformers Val Acc: {transformers_results["eval_accuracy"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison.png'))
    plt.show()

def main():
    # Hàm chính thực thi chương trình
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Thiết lập thiết bị: {device} ===")
    
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tiền xử lý dữ liệu
    print("\n=== Đang tải và tiền xử lý dữ liệu ===")
    df = pd.read_excel(args.data, sheet_name='Sheet1')
    df['text'] = df['Câu'].astype(str).str.lower()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Nhãn cảm xúc'])
    
    # Chia tập train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values, df['label'].values, test_size=0.2, stratify=df['label'], random_state=42)
    
    # 1. Huấn luyện mô hình Transformers
    transformers_model, transformers_results = train_transformers_model(
        X_train, y_train, X_val, y_val, args, le)
    
    # 2. Huấn luyện mô hình LSTM tự xây dựng
    # Chuẩn bị dữ liệu cho LSTM
    tokenized = [s.split() for s in X_train]
    freq = Counter(chain.from_iterable(tokenized))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in freq.most_common(args.max_vocab):
        vocab[word] = len(vocab)
    
    def encode_and_pad(sentences, max_len=None):
        encoded = [[vocab.get(w, vocab['<UNK>']) for w in s.split()] for s in sentences]
        if max_len is None:
            max_len = max(len(s) for s in encoded)
        lengths = torch.tensor([min(len(s), max_len) for s in encoded])
        padded = [s[:max_len] + [vocab['<PAD>']] * (max_len - len(s)) if len(s) < max_len else s[:max_len] for s in encoded]
        return torch.tensor(padded), lengths
    
    X_train_tensor, len_train = encode_and_pad(X_train)
    X_val_tensor, len_val = encode_and_pad(X_val, max_len=X_train_tensor.shape[1])
    y_train_tensor = torch.tensor(y_train)
    y_val_tensor = torch.tensor(y_val)
    
    custom_results = train_custom_model(
        X_train_tensor, len_train, y_train_tensor,
        X_val_tensor, len_val, y_val_tensor,
        vocab, le, args, device
    )
    
    # 3. So sánh kết quả
    plot_comparison(transformers_results, custom_results, args)
    
    # Tổng hợp kết quả
    print("\n=== Kết quả tổng hợp ===")
    print(f"Transformers Model - Val Loss: {transformers_results['eval_loss']:.4f}, Val Acc: {transformers_results['eval_accuracy']:.4f}")
    print(f"Custom LSTM Model - Best Val Loss: {min(custom_results['val_losses']):.4f}, Best Val Acc: {max(custom_results['val_accs']):.4f}")

    # Giải thích mô hình và thuật toán
    print("\n=== Giải thích mô hình và thuật toán ===")
    print("""
1. Mô hình Transformers (PhoBERT):
- Sử dụng kiến trúc Transformer tiền huấn luyện trên tiếng Việt
- Tự động học các đặc trưng ngữ nghĩa từ dữ liệu lớn
- Hiệu quả với các tác vụ NLP như phân loại cảm xúc

2. Mô hình LSTM tự xây dựng:
- Kiến trúc Bidirectional LSTM (BiLSTM) 4 lớp
- Cơ chế Attention giúp tập trung vào từ quan trọng
- Quy trình huấn luyện:
  * Embedding từ → BiLSTM → Attention → Fully-connected
  * Tối ưu bằng AdamW với learning rate scheduling
  * Loss function: Cross-entropy

3. So sánh:
- Transformers thường mạnh hơn nhưng yêu cầu tài nguyên lớn
- LSTM nhẹ hơn, phù hợp khi dữ liệu ít hoặc phần cứng hạn chế
- LSTM có thể hiệu quả với các tác vụ đơn giản hoặc dữ liệu đặc thù
""")

if __name__ == '__main__':
    main()