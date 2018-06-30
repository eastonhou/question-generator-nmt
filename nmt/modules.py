import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim


    def forward(self, emb):
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:emb.size(0)]
        emb = self.dropout(emb)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1E-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps


    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2*(x-mean)/(std+self.eps) + self.b2

        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, hidden_size, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, dim)
        self.layer_norm = LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        inter = self.dropout1(self.relu(self.w1(self.layer_norm(x))))
        output = self.dropout2(self.w2(inter))
        return output + x