import torch
import torch.nn as nn

class MyClassifier( nn.Module ):
    def __init__(self, vocab_size, embedding_dim, output_dim, n_special_tokens=2):
        super(MyClassifier, self).__init__()
        self.n_special_tokens = n_special_tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self._pool(x)
        x = self.fc(x)
        return x