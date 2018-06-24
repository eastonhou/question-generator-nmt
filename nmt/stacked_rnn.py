import torch
import torch.nn as nn

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size


    def forward(self, input, hidden):
        h0, c0 = hidden
        h1, c1 = [], []
        for i, layer in enumerate(self.layers):
            h1i, c1i = layer(input, (h0[i], c0[i]))
            input = h1i
            if i+1 != self.num_layers:
                input = self.dropout(input)
            h1.append(h1i)
            c1.append(c1i)
        h1 = torch.stack(h1)
        c1 = torch.stack(c1)
        hidden = (h1, c1)
        return input, hidden