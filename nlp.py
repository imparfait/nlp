from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from collections import Counter
import re

dataset = load_dataset("emotion")
print(dataset["train"][0])

def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

def build_vocab(dataset, min_freq=2):
    freq = Counter()
    for item in dataset:
        tokens = tokenize(item["text"])
        freq.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=30):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    ids = ids[:max_len]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

class EmotionDataset(Dataset):
        def __init__(self, data, vocab, label2idx, max_len=30):
            self.texts = [encode(item["text"], vocab, max_len) for item in data]
            self.labels = [label2idx[dataset["train"].features["label"].int2str(item["label"])] for item in data]
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])
        
selected_labels = ["joy", "anger", "sadness"]
filtered_train = [item for item in dataset["train"] if item["label"] in [dataset["train"].features["label"].names.index(l) for l in selected_labels]]
filtered_test = [item for item in dataset["test"] if item["label"] in [dataset["test"].features["label"].names.index(l) for l in selected_labels]]

label2idx = {"joy": 0, "anger": 1, "sadness": 2}
idx2label = {v: k for k, v in label2idx.items()}
vocab = build_vocab(filtered_train)
train_dataset = EmotionDataset(filtered_train, vocab, label2idx)
test_dataset = EmotionDataset(filtered_test, vocab, label2idx)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class EmotionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out
    
VOCAB_SIZE = max(vocab.values()) + 1
EMBED_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 3
model = EmotionRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
class_names = list(label2idx.keys())  # ['joy', 'anger', 'sadness']
train_labels = [dataset["train"].features["label"].int2str(item["label"]) for item in filtered_train]

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(class_names),
    y=train_labels
)
ordered_weights = [class_weights[class_names.index(label)] for label in label2idx.keys()]
class_weights_tensor = torch.tensor(ordered_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
all_preds, all_labels = [], []
    
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
print(classification_report(all_labels, all_preds, target_names=label2idx.keys()))

def predict(text):
    model.eval()
    encoded = encode(text, vocab)
    tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return idx2label[pred]
print("Train label counts:", Counter(train_labels))
print(predict("I am so happy today!"))
print(predict("this makes me angry"))
print(predict("i feel very sad")) 