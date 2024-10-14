import torch
import torch.nn as nn
import torch.nn.functional as F
import my_nlp_library as nlp
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
            self.hidden_layers.append(nn.Dropout(p=0.1))
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
    

class MyMLPResidualNetworkWithGloveEmbeddings( nn.Module ):
    def __init__(self, hidden_dim, glove_vectors, output_dim, n_hidden_layers=1, n_special_tokens=2):
        super(MyMLPResidualNetworkWithGloveEmbeddings, self).__init__()
        self.n_special_tokens = n_special_tokens
        vocab, inverse_vocab = nlp.get_vocabulary_from_glove(glove_vectors)
        embedding = nlp.make_embedding_layer_from_glove(glove_vectors, vocab, inverse_vocab, 300)
        self.embedding = embedding
        self.mlp = ResidualMLP(300, hidden_dim, n_hidden_layers, output_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self._pool(x)
        x = self.mlp(x)
        return x


class MyMLPResidualNetworkWithGloveEmbeddingsRNN( nn.Module ):
    def __init__(self, hidden_dim, glove_vectors, output_dim, n_layers_rnn=1, n_hidden_layers_mlp=1, n_special_tokens=2):
        super(MyMLPResidualNetworkWithGloveEmbeddingsRNN, self).__init__()
        self.n_special_tokens = n_special_tokens
        vocab, inverse_vocab = nlp.get_vocabulary_from_glove(glove_vectors)
        embedding = nlp.make_embedding_layer_from_glove(glove_vectors, vocab, inverse_vocab, 300)
        self.sequence_model = nn.RNN(300, hidden_dim, n_layers_rnn, batch_first=True)
        self.embedding = embedding
        self.mlp = ResidualMLP(hidden_dim, hidden_dim, n_hidden_layers_mlp, output_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def _pool(self, x):
        _, x = self.sequence_model(x)
        x = x.reshape(x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = self._pool(x)
        x = self.mlp(x)
        return x


class MyMLPResidualNetworkWithGloveEmbeddingsLSTMMeanPooling( nn.Module ):
    def __init__(self, hidden_dim, glove_vectors, output_dim, n_layers_rnn=1, n_hidden_layers_mlp=1, n_special_tokens=2):
        super(MyMLPResidualNetworkWithGloveEmbeddingsLSTMMeanPooling, self).__init__()
        self.n_special_tokens = n_special_tokens
        vocab, inverse_vocab = nlp.get_vocabulary_from_glove(glove_vectors)
        embedding = nlp.make_embedding_layer_from_glove(glove_vectors, vocab, inverse_vocab, 300)
        self.sequence_model = nn.LSTM(300, hidden_dim, n_layers_rnn, batch_first=True)
        self.embedding = embedding
        self.mlp = ResidualMLP(hidden_dim, hidden_dim, n_hidden_layers_mlp, output_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def _pool(self, x):
        x = torch.mean(x, dim=1)
#        x = x.reshape(x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.sequence_model(x)
        x = self._pool(x)
        x = self.mlp(x)
        return x

class MyMLPResidualNetworkWithGloveEmbeddingsLSTMLastState( nn.Module ):
    def __init__(self, hidden_dim, glove_vectors, output_dim, n_layers_rnn=1, n_hidden_layers_mlp=1, n_special_tokens=2):
        super(MyMLPResidualNetworkWithGloveEmbeddingsLSTMLastState, self).__init__()
        self.n_special_tokens = n_special_tokens

        vocab, inverse_vocab = nlp.get_vocabulary_from_glove(glove_vectors)
        embedding = nlp.make_embedding_layer_from_glove(glove_vectors, vocab, inverse_vocab, 300)
        self.sequence_model = nn.LSTM(300, hidden_dim, n_layers_rnn, batch_first=True)
        self.embedding = embedding
        self.mlp = ResidualMLP(hidden_dim, hidden_dim, n_hidden_layers_mlp, output_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.embedding(x)
        _, (_, x) = self.sequence_model(x)
        x = x.reshape(x.shape[1], x.shape[2])
        x = self.mlp(x)
        return x


class MyMLPResidualNetworkWithGloveEmbeddingsLSTMLastHidden( nn.Module ):
    def __init__(self, hidden_dim, glove_vectors, output_dim, n_layers_rnn=1, n_hidden_layers_mlp=1, n_special_tokens=2):
        super(MyMLPResidualNetworkWithGloveEmbeddingsLSTMLastHidden, self).__init__()
        self.n_special_tokens = n_special_tokens

        vocab, inverse_vocab = nlp.get_vocabulary_from_glove(glove_vectors)
        embedding = nlp.make_embedding_layer_from_glove(glove_vectors, vocab, inverse_vocab, 300)
        self.sequence_model = nn.LSTM(300, hidden_dim, n_layers_rnn, batch_first=True)
        self.embedding = embedding
        self.mlp = ResidualMLP(hidden_dim, hidden_dim, n_hidden_layers_mlp, output_dim)
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.embedding(x)
        _, (x, h) = self.sequence_model(x)
        x = x.reshape(x.shape[1], x.shape[2])
        x = self.mlp(x)
        return x
    


class MyMultiHeadAttention( nn.Module ):
    def __init__(self, embedding_dim, n_attention_heads):
        super(MyMultiHeadAttention, self).__init__()
        self.n_attention_heads = n_attention_heads
        self.mha = nn.MultiheadAttention(embedding_dim, n_attention_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, v, k, q):
        res = q
        x, _ = self.mha(v, k, q)
        x = x + res
        x = self.norm(x)
        return x

class MyResidualNetworkWithMultiHeadAttention( nn.Module ):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, seq_len=500, n_hidden_layers=1, n_attention_heads=8, n_special_tokens=2):
        super(MyResidualNetworkWithMultiHeadAttention, self).__init__()
        self.n_special_tokens = n_special_tokens
        self.positional_encoding = nn.Embedding(seq_len, embedding_dim)
        self.mha = MyMultiHeadAttention(embedding_dim, n_attention_heads)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = ResidualMLP(embedding_dim, hidden_dim, n_hidden_layers, output_dim)
        

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def forward(self, x):
        x = self.embedding(x)
        pos_encodings = self.positional_encoding(torch.tensor(range(x.shape[1])).to(next(self.parameters()).device))
        x = x + pos_encodings
        x = self.mha(x, x, x)
        x = self._pool(x)
        x = self.mlp(x)
        return x
    
