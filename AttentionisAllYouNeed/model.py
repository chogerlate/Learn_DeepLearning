import torch
import numpy as np
import torch.nn as nn
import math

class ImputEmbeddings(nn.Module):
    
    def __init__(self, vocab_size, d_model):
        super(ImputEmbeddings, self).__init__()
        self.d_model = d_model # dimentions of the embedding vector
        self.vocab_size = vocab_size # size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model) # embedding layer
        # simpily, mapping number to vector size of d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model # dimentions of the embedding vector
        self.seq_len = seq_len # maximum length of the sequence
        self.dropout = nn.Dropout(dropout)
        
        # create constant 'pe' matrix with values dependant on (pos, i)
        pe = torch.zeros(seq_len, d_model)
        
        # calculate the positional encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # apply the cosine to even columns and sin to odds
        pe[:, 0::2] = torch.sin(position * div_term) # (seq_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term) # (seq_len, d_model)
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
    
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Mulitplicative alpha parameter
        self.bias = nn.Parameter(torch.zeros(1)) # Additive bias parameter
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # mean of the last dimention
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
