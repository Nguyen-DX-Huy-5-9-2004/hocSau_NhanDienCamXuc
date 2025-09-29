import os
import pickle
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import pytorch_lightning as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Thiết bị kiểm tra: {device}")

def load_required_files():
    required_files = {
        'vocab.pkl': 'Từ điển từ vựng',
        'max_len.pkl': 'Độ dài tối đa câu',
        'label_encoder_classes.npy': 'Nhãn cảm xúc',
        'best_emotion_model.pt': 'Mô hình thủ công có Attention',
        'best_emotion_model_no_attn.pt': 'Mô hình thủ công không Attention',
        'emotion_model_lightning.pt': 'Mô hình thư viện'
    }
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\nLỖI: Thiếu các file cần thiết để chạy kiểm tra:")
        for file in missing_files:
            print(f"- {file} ({required_files[file]})")
        print("\nHãy chạy lại chương trình huấn luyện để tạo các file này trước!")
        exit()
    
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    with open('max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder_classes.npy', allow_pickle=True)
    
    return vocab, max_len, label_encoder

vocab, max_len, label_encoder = load_required_files()
num_classes = len(label_encoder.classes_)
pad_idx = vocab['<PAD>']

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

class EmotionClassifier(pl.LightningModule):
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
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

manual_model = EmotionBiLSTM(len(vocab), 128, 128, num_classes, pad_idx).to(device)
manual_model.load_state_dict(torch.load('best_emotion_model.pt', map_location=device))
manual_model.eval()

no_attn_model = EmotionBiLSTM_NoAttention(len(vocab), 128, 128, num_classes, pad_idx).to(device)
no_attn_model.load_state_dict(torch.load('best_emotion_model_no_attn.pt', map_location=device))
no_attn_model.eval()

lightning_model = EmotionClassifier(len(vocab), 128, 128, num_classes, pad_idx).to(device)
lightning_model.load_state_dict(torch.load('emotion_model_lightning.pt', map_location=device))
lightning_model.eval()

def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    if not sentence:
        return None
    
    encoded = [vocab.get(tok, vocab['<UNK>']) for tok in sentence.split()]
    
    if len(encoded) < max_len:
        padded = encoded + [vocab['<PAD>']] * (max_len - len(encoded))
    else:
        padded = encoded[:max_len]
    return torch.tensor([padded], dtype=torch.long).to(device)

def predict_with_all_models(sentence):
    input_tensor = preprocess_sentence(sentence)
    if input_tensor is None:
        return None
    
    with torch.no_grad():
        manual_output = manual_model(input_tensor)
        manual_probs = F.softmax(manual_output, dim=1)
        manual_prob, manual_label = torch.max(manual_probs, dim=1)
        
        no_attn_output = no_attn_model(input_tensor)
        no_attn_probs = F.softmax(no_attn_output, dim=1)
        no_attn_prob, no_attn_label = torch.max(no_attn_probs, dim=1)
        
        lightning_output = lightning_model(input_tensor)
        lightning_probs = F.softmax(lightning_output, dim=1)
        lightning_prob, lightning_label = torch.max(lightning_probs, dim=1)
    
    manual_emotion = label_encoder.inverse_transform([manual_label.item()])[0]
    no_attn_emotion = label_encoder.inverse_transform([no_attn_label.item()])[0]
    lightning_emotion = label_encoder.inverse_transform([lightning_label.item()])[0]
    
    results = {
        'manual_attn': {
            'emotion': manual_emotion,
            'confidence': manual_prob.item(),
            'prob_dist': manual_probs.squeeze().cpu().numpy()
        },
        'manual_no_attn': {
            'emotion': no_attn_emotion,
            'confidence': no_attn_prob.item(),
            'prob_dist': no_attn_probs.squeeze().cpu().numpy()
        },
        'lightning': {
            'emotion': lightning_emotion,
            'confidence': lightning_prob.item(),
            'prob_dist': lightning_probs.squeeze().cpu().numpy()
        }
    }
    
    return results

def print_comparison(results):
    print(f"\n{'Mô hình':<20} | {'Cảm xúc':<15} | {'Độ tin cậy':<15}")
    print("-" * 60)
    print(f"{'Thủ công có Attention':<20} | {results['manual_attn']['emotion']:<15} | {results['manual_attn']['confidence']*100:.2f}%")
    print(f"{'Thủ công không Attention':<20} | {results['manual_no_attn']['emotion']:<15} | {results['manual_no_attn']['confidence']*100:.2f}%")
    print(f"{'Thư viện Lightning':<20} | {results['lightning']['emotion']:<15} | {results['lightning']['confidence']*100:.2f}%")
    
    print("\nPhân phối xác suất (top 3):")
    for model_name, model_label in [('manual_attn', 'Thủ công có Attention'), 
                                  ('manual_no_attn', 'Thủ công không Attention'), 
                                  ('lightning', 'Thư viện Lightning')]:
        print(f"\n{model_label}:")
        probs = results[model_name]['prob_dist']
        top3_indices = np.argsort(probs)[-3:][::-1]
        for idx in top3_indices:
            emotion = label_encoder.inverse_transform([idx])[0]
            print(f"- {emotion}: {probs[idx]*100:.2f}%")

print(f"\nDanh sách cảm xúc cs thể dự đoán: {list(label_encoder.classes_)}")

while True:
    sentence = input("\nNhập câu cần phân tích cảm xúc (hoặc 'exit' để thoát): ")
    if sentence.lower() == 'exit':
        break
    
    results = predict_with_all_models(sentence)
    
    if results is None:
        print("LỖI: Câu nhập vào không hợp lệ!")
        continue
    
    print(f"\nKết quả phân tích cho câu: '{sentence}'")
    print_comparison(results)