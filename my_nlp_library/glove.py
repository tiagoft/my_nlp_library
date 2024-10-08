import urllib.request
import zipfile
import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn

target_url = "http://nlp.stanford.edu/data/glove.6B.zip"
output = "glove.6B.zip"
txt_filename = "glove.6B.300d.txt"
cachedir = Path (os.path.expanduser('~/.my_nlp_library'))
path_to_glove_embeddings = cachedir / txt_filename

class ZipFileWithProgress(zipfile.ZipFile):
        def extractall(self, path=None, members=None, pwd=None):
            if members is None:
                members = self.namelist()
            total = len(members)
            
            with tqdm(total=total, unit='file') as pbar:
                for member in members:
                    self.extract(member, path, pwd)
                    pbar.update(1)

def get_glove():
    def download_progress(block_num, block_size, total_size):
        progress = block_num * block_size / total_size * 100
        print(f"\rDownloading: {progress:.2f}%", end='')


    local_file_path = cachedir / output

    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    
    if not os.path.exists(local_file_path):
        urllib.request.urlretrieve(target_url, local_file_path, reporthook=download_progress)

    
    if not os.path.exists(path_to_glove_embeddings):
        # Use the custom ZipFileWithProgress class
        with ZipFileWithProgress(local_file_path, 'r') as zip_ref:
            zip_ref.extractall(cachedir)

    return path_to_glove_embeddings

def load_glove_vectors(glove_file=path_to_glove_embeddings, download=True):
    if download==True:
        get_glove()

    glove_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            glove_vectors[word] = vector
    return glove_vectors

def get_vocabulary_from_glove(glove_vectors):
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for word, vector in glove_vectors.items():
        vocab[word] = len(inverse_vocab)
        inverse_vocab.append(word)
    return vocab, inverse_vocab

def make_embedding_layer_from_glove(glove_vectors, vocab, inverse_vocab, embedding_dim=300):
    vocab_size = len(glove_vectors) + 2
    embedding = nn.Embedding(vocab_size, embedding_dim)

    for idx, word in enumerate(inverse_vocab[2:]):
        i = idx + 2
        embedding.weight[idx].data = glove_vectors[word]
    return embedding

