import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import my_nlp_library as nlp
import os
import urllib.request
from pathlib import Path

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        x = torch.tensor(tokens)
        y = torch.tensor(label).reshape(1).clone().detach()
        return x, y
    
def get_imdb_dataset(target_url : str='https://raw.githubusercontent.com/tiagoft/NLP/refs/heads/main/Aulas/datasets/IMDB%20Dataset.csv',
                       sentence_length : int = 100,
                       sample_size : int=None):
    cachedir = Path (os.path.expanduser('~/.my_nlp_library'))
    local_file_path = cachedir / 'IMDB_Dataset.csv'
    
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    
    if not os.path.exists(local_file_path):
        urllib.request.urlretrieve(target_url, local_file_path)
    
    df = pd.read_csv(local_file_path).sample(sample_size)
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
    classes = list(set(y_train))
    y_train_bin = torch.tensor([[classes.index(y) for y in y_train]]).T
    y_test_bin = torch.tensor([[classes.index(y) for y in y_test]]).T

    tokenizer = nlp.MyTokenizer(sentence_length=sentence_length)
    tokenizer.fit(X_train)

    dataset_train = TextDataset(list(X_train), y_train_bin, tokenizer)
    dataset_test = TextDataset(list(X_test), y_test_bin, tokenizer)

    return dataset_train, dataset_test, tokenizer