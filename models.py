import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import nmt
import nmt.stacked_rnn as stacked_rnn
import nmt.attention as attention
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class DecoderState(object):
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.shape
            br = sizes[1]
            if e.dim() == 3:
                sent_states = e.view(sizes[0], beam_size, br//beam_size, sizes[2])[:,:,idx]
            else:
                sent_states = e.view(sizes[0], beam_size, br//beam_size, sizes[2], sizes[3])[:,:,idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, hidden_size, rnnstate):
        self.hidden = rnnstate if isinstance(rnnstate, tuple) else (rnnstate,)
        batch_size = self.hidden[0].shape[1]
#        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].data.new(batch_size, hidden_size).zero_().unsqueeze(0)


    @property
    def _all(self):
        return self.hidden + (self.input_feed,)


    def update_state(self, rnnstate, input_feed, coverage):
        self.hidden = rnnstate if isinstance(rnnstate, tuple) else (rnnstate,)
        self.input_feed = input_feed
        self.coverage = coverage


    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [e.data.repeat(1, beam_size, 1) for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class EncoderBase(nn.Module):
    def forward(self, src, lengths=None, encoder_state=None):
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    def __init__(self, num_layers, vocab_size, embedding_dim, hidden_size, bidirectional, dropout, use_bridge=False):
        super(RNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=data.NULL_ID)
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)


    def forward(self, src, lengths, encoder_state=None):
        emb = self.embeddings(src)
        if lengths is None:
            lengths = (src != data.NULL_ID).sum(-1)
        packed_emb = pack(emb, lengths)
        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        memory_bank = unpack(memory_bank)[0]
        return encoder_final, memory_bank


class RNNDecoderBase(nn.Module):
    def __init__(self, embeddings, num_layers, bidirectional_encoder, hidden_size, attn_type, dropout):
        super(RNNDecoderBase, self).__init__()
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.rnn = stacked_rnn.StackedLSTM(num_layers, self._input_size, hidden_size, dropout)
        self.attn = attention.GlobalAttention(hidden_size, attn_type=attn_type)
        self.bidirectional_encoder = bidirectional_encoder
        self.dropout = nn.Dropout(dropout)


    @property
    def _input_size(self):
        return self.embeddings.weight.shape[1]


    def forward(self, tgt, memory_bank, state, memory_lengths):
        decoder_final, decoder_outputs, attns = self._run_forward_pass(tgt, memory_bank, state, memory_lengths=memory_lengths)
        final_output = decoder_outputs[-1]
        state.update_state(decoder_final, final_output.unsqueeze(0), None)
        if not isinstance(decoder_outputs, torch.Tensor):
            decoder_outputs = torch.stack(decoder_outputs)
        for key in attns:
            if not isinstance(attns[key], torch.Tensor):
                attns[key] = torch.stack(attns[key])
        return decoder_outputs, state, attns


    def init_decoder_state(self, src, memory_bank, encoder_final):
        def _fix_enc_hidden(h):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h
        if isinstance(encoder_final, tuple): #LSTM
            return RNNDecoderState(self.hidden_size, tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final]))
        else: #GRU
            return RNNDecoderState(self.hidden_size, _fix_enc_hidden(encoder_final))


class InputFeedRNNDecoder(RNNDecoderBase):
    @property
    def _input_size(self):
        return self.embeddings.weight.shape[1] + self.hidden_size


    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths):
        input_feed = state.input_feed.squeeze(0)
        decoder_outputs = []
        attns = {'std': []}
        emb = self.embeddings(tgt)#[time, batch, dim]
        hidden = state.hidden
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)#[batch, dim]
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(rnn_output, memory_bank.transpose(0, 1), memory_lengths=memory_lengths)
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            decoder_outputs.append(decoder_output)
            attns['std'].append(p_attn)
        return hidden, decoder_outputs, attns


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        vocab_size = self.decoder.embeddings.weight.shape[0]
        self.generator = nn.Sequential(nn.Linear(self.decoder.hidden_size, vocab_size), nn.LogSoftmax(dim=-1))
        self.generator[0].weight = self.decoder.embeddings.weight


    def forward(self, src, tgt, lengths, dec_state=None):
        tgt = tgt[:-1]
        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank, enc_state if dec_state is None else dec_state, memory_lengths=lengths)
        slen, batch_size, embedding_dim = decoder_outputs.shape
        decoder_outputs = self.generator(decoder_outputs.view(-1, embedding_dim)).view(slen, batch_size, -1)
        return decoder_outputs, attns, dec_state



def make_loss_compute(vocab_size):
    weight = torch.ones(vocab_size)
    weight[data.NULL_ID] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    return criterion


def build_model(opt, vocab_size):
    encoder = RNNEncoder(opt.num_layers, vocab_size, opt.word_vec_size, opt.rnn_size, opt.bidirectional_encoder, opt.dropout)
    decoder = InputFeedRNNDecoder(encoder.embeddings, opt.num_layers, opt.bidirectional_encoder, opt.rnn_size, opt.attn_type, opt.dropout)
    model = NMTModel(encoder, decoder)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


if __name__ == '__main__':
    encoder = RNNEncoder(2, 10000, 512, 512, True, 0.3, True)
