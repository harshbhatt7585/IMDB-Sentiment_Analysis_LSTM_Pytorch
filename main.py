import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer 
# from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import Counter
import pandas as pd
import os
from torchtext.datasets import IMDB
from torchtext import data
import random
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset

class TextDataset(Dataset):
    def __init__(self, X, y, dataset_type='train'):
        len_ = len(X)
        self.X = X
        self.y = torch.tensor(y)

        if(dataset_type == 'train'):
            self.X = self.X[: int(len_ * 0.8)]
            self.y = self.y[: int(len_ * 0.8)]
        else:
            self.X = self.X[int(len_ * 0.8): ]
            self.y = self.y[int(len_ * 0.8): ]
        
        self.size = len(self.X)

    
    
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.size  


class Classifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, output_dim=1):
        super(Classifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)   
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.embeddings(x)
        # x, (h, c) = self.lstm(x)
        # x = self.dropout(x) 
        x = self.dense1(x)
        x = self.sigmoid(x)
        return x  

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
    

if __name__ == '__main__':
    BATCH_SIZE = 128
    EPOCHS = 10
    device = torch.device("cpu")
    
    train_dataset = IMDB(split='train')

    tokenizer = get_tokenizer('spacy') 
    vocab = build_vocab_from_iterator(yield_tokens(train_dataset), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])  

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    train_dataset = to_map_style_dataset(train_dataset)
    test_dataset = to_map_style_dataset(train_dataset)
    num_train = int(len(train_dataset) * 0.8)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_loader = DataLoader(split_train_, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    val_loader = DataLoader(split_valid_, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    vocab_size = len(vocab)
    model = Classifier(vocab_size).to(device)
    
    criterion = nn.BCELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(EPOCHS):
        train_loss = 0
        train_acc = 0
        model.train()
        for step, batch in enumerate(train_loader):
            optim.zero_grad()
            x = batch[1]
            y = batch[0].float()
            y_pred = torch.squeeze(model(x), 1)
            loss = criterion(y_pred, y)
          
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optim.step()
            acc = ((y_pred.round() == y).sum()/BATCH_SIZE).item()
            print(loss)
            train_acc += acc
            train_loss += loss.item()
        print("Train Loss: ", train_loss / len(iter(train_loader)))
        print("Train Acc: ", train_acc / len(iter(train_loader)))

        val_acc = 0
    
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x = batch[1]
                y = batch[0].float()
                y_pred = torch.squeeze(model(x), 1)
                acc = ((y_pred.round() == y).sum()/BATCH_SIZE).item()
                val_acc += acc
            print("-------------------------------------------------------")
            print("Val Acc: ", val_acc / len(iter(val_loader)))
            print("-------------------------------------------------------")
