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
    

# Base MLP types
class MLP (nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_hidden_layers,
                 output_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
        self.output_layer= nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    

class MyNetwork( nn.Module ):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_hidden_layers=1, n_special_tokens=2):
        super(MyNetwork, self).__init__()
        self.n_special_tokens = n_special_tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = MLP(embedding_dim, hidden_dim, n_hidden_layers, output_dim)

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self._pool(x)
        x = self.mlp(x)
        return x
    

# Base residual MLPs
class ResidualModule(nn.Module):
    def __init__(self, dim):
        super(ResidualModule, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x + res
        return x

class ResidualMLP (nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 n_hidden_layers,
                 output_dim):
        super(ResidualMLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(ResidualModule(hidden_dim))
        self.output_layer= nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class MyResidualNetwork( nn.Module ):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_hidden_layers=1, n_special_tokens=2):
        super(MyResidualNetwork, self).__init__()
        self.n_special_tokens = n_special_tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = ResidualMLP(embedding_dim, hidden_dim, n_hidden_layers, output_dim)

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self._pool(x)
        x = self.mlp(x)
        return x
    
