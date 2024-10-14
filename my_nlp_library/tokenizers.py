import re

def get_vocabulary(text : str,
                   expr: str=r"\b\w+\b",
                   case_sensitive : bool=False,
                   ) -> dict:
    if case_sensitive == False:
        text = text.lower()  
    vocabulary = set(re.findall(expr, text))
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for i, token in enumerate(vocabulary):
        if token not in vocab:
            vocab[token] = i+2 # We start from 2 because 0 and 1 are reserved for <UNK> and <PAD>
            inverse_vocab.append(token)
    return vocab, inverse_vocab

def tokenize_words(text : str,
             vocab : dict,
             expr : str= r"\b\w+\b",
             sentence_length : int = 10,
             case_sensitive : bool = False) -> list:
    if case_sensitive == False:
        text = text.lower()
    words = re.findall(expr, text)
    tokens = []
    for i, w in enumerate(words):
        if i == sentence_length:
            break
        if w in vocab:
            tokens.append(vocab[w])
        else:
            tokens.append(vocab["<UNK>"])


    if len(tokens) < sentence_length:
        n_pad = sentence_length - len(tokens)
        pad = [vocab["<PAD>"]] * n_pad
        tokens = pad + tokens
    return tokens

def detokenize_words(tokens : list,
                    invert_vocab : list) -> str:
    text = " ".join([invert_vocab[token] for token in tokens])
    return text    

class MyTokenizer:
    def __init__(self, sentence_length, case_sensitive=False, vocab=None, inverse_vocab=None):
        self.sentence_length = sentence_length
        self.case_sensitive = case_sensitive
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        if vocab is not None:
            self.vocab_size = len(vocab)

    def fit(self, phrases : list, expr : str=r"\b\w+\b"):
        self.vocab, self.inverse_vocab = get_vocabulary(" ".join(phrases),
                                                        expr=expr,
                                                        case_sensitive=self.case_sensitive)
        self.vocab_size = len(self.vocab)
        
    def __call__(self, x):
        return tokenize_words(x,
                              self.vocab,
                              sentence_length=self.sentence_length,
                              case_sensitive=self.case_sensitive)
    

