import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import nmt
import nmt.stacked_rnn as stacked_rnn
import nmt.attention as attention
import nmt.modules as modules
import nmt.utils as nu
import numpy as np
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


class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None


    @property
    def _all(self):
        return self.previous_input, self.previous_layer_inputs, self.src


    def update_state(self, input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = input
        state.previous_layer_inputs = previous_layer_inputs
        return state


    def repeat_beam_size_times(self, beam_size):
        self.src = self.src.data.repeat(1, beam_size, 1)


class EncoderBase(nn.Module):
    def forward(self, src, lengths=None, encoder_state=None):
        raise NotImplementedError


class RNNEncoder(EncoderBase):
    def __init__(self, embeddings, num_layers, hidden_size, bidirectional, dropout, use_bridge=False):
        super(RNNEncoder, self).__init__()
        self.embeddings = embeddings
        self.vocab_size, self.embedding_dim = embeddings[0].weight.shape
        hidden_size = hidden_size//2 if bidirectional else hidden_size
        self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)


    def forward(self, src, lengths, encoder_state=None):
        emb = self.embeddings(src)
        if lengths is None:
            lengths = (src != data.NULL_ID).sum(-1)
        packed_emb = pack(emb, lengths)
        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        memory_bank = unpack(memory_bank)[0]
        return encoder_final, memory_bank


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, head_count, hidden_size, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = attention.MultiHeadedAttention(head_count, dim, dropout)
        self.feed_forward = modules.PositionwiseFeedForward(dim, hidden_size, dropout)
        self.layer_norm = modules.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, embeddings, num_layers, head_count, hidden_size, dropout):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        _, dim = embeddings[0].weight.shape
        self.transformer = nn.ModuleList([TransformerEncoderLayer(dim, head_count, hidden_size, dropout) for _ in range(num_layers)])
        self.layer_norm = modules.LayerNorm(dim)


    def forward(self, input, lengths, hidden=None):
        emb = self.embeddings(input)
        out = emb.transpose(0, 1).contiguous()#[batch, plen, dim]
        words = input.transpose(0, 1)#[batch, plen]
        batch_size, plen = words.shape
        mask = words.eq(data.NULL_ID).unsqueeze(1).expand(batch_size, plen, plen)
        for m in self.transformer:
            out = m(out, mask)
        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous()#[plen, batch, dim], [plen, batch, dim]


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
        return self.embeddings[0].weight.shape[1]


    def forward(self, tgt_or_generator, memory_bank, state, memory_lengths):
        if isinstance(tgt_or_generator, nn.Sequential):
            _forward = self._run_free_pass
        else:
            _forward = self._run_forward_pass
        decoder_final, decoder_outputs, attns = _forward(tgt_or_generator, memory_bank, state, memory_lengths=memory_lengths)
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
        return self.embeddings[0].weight.shape[1] + self.hidden_size


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


    def _run_free_pass(self, generator, memory_bank, state, memory_lengths):
        input_feed = state.input_feed.squeeze(0)
        decoder_outputs = []
        attns = {'std': []}
        batch_size = memory_lengths.shape[0]
        batch_sos = nu.tensor([[data.SOS_ID]*batch_size])
        emb_t = self.embeddings(batch_sos).squeeze(0)#[batch, dim]
        hidden = state.hidden
        for _ in range(20):
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(rnn_output, memory_bank.transpose(0, 1), memory_lengths=memory_lengths)
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            decoder_outputs.append(decoder_output)
            attns['std'].append(p_attn)
            _, ids = generator(decoder_output).max(-1)
            if ids.eq(data.NULL_ID).sum().tolist() == batch_size:
                break
            emb_t = self.embeddings(ids.unsqueeze(0)).squeeze(0)
        return hidden, decoder_outputs, attns


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, head_count, hidden_size, dropout, max_size=400):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = attention.MultiHeadedAttention(head_count, dim, dropout)
        self.context_attn = attention.MultiHeadedAttention(head_count, dim, dropout)
        self.feed_forward = modules.PositionwiseFeedForward(dim, hidden_size, dropout)
        self.layer_norm1 = modules.LayerNorm(dim)
        self.layer_norm2 = modules.LayerNorm(dim)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(max_size)
        self.register_buffer('mask', mask)


    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=None):
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.shape[1], :tgt_pad_mask.shape[1]], 0)
        input_norm = self.layer_norm1(inputs)
        all_input = input_norm#[batch, plen, dim]
        if previous_input is not None:
            all_input = torch.cat([previous_input, input_norm], dim=1)
            dec_mask = None
        query, attn = self.self_attn(all_input, all_input, input_norm, mask=dec_mask)
        query = self.drop(query) + inputs
        query_norm = self.layer_norm2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)
        return output, attn, all_input


    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

        
class TransformerDecoder(nn.Module):
    def __init__(self, embeddings, num_layers, head_count, hidden_size, dropout):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        dim = embeddings[0].weight.shape[1]
        self.transformer_layers = nn.ModuleList([TransformerDecoderLayer(dim, head_count, hidden_size, dropout) for _ in range(num_layers)])
        self.layer_norm = modules.LayerNorm(dim)


    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return TransformerDecoderState(src)

        
    def forward(self, tgt, memory_bank, state, memory_lengths):
        assert isinstance(state, TransformerDecoderState)
        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        outputs = []
        attns = {'std': []}
        emb = self.embeddings(tgt)#[plen, batch, dim]
        if state.previous_input is not None:
            emb = emb[state.previous_input.shape[0]:,]
        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        src = state.src
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        batch_size, src_len = src_words.shape
        _, tgt_len = tgt_words.shape
        src_pad_mask = src_words.data.eq(data.NULL_ID).unsqueeze(1).expand(batch_size, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(data.NULL_ID).unsqueeze(1).expand(batch_size, tgt_len, tgt_len)
        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input = self.transformer_layers[i](output, src_memory_bank, src_pad_mask, tgt_pad_mask, previous_input=prev_layer_input)
            saved_inputs.append(all_input)
        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)
        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn
        # Update the state.
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        vocab_size, dim = self.decoder.embeddings[0].weight.shape
        self.generator = nn.Sequential(nn.Linear(dim, vocab_size), nn.LogSoftmax(dim=-1))
        self.generator[0].weight = self.decoder.embeddings[0].weight


    def forward(self, src, tgt, lengths, dec_state=None):
        tgt = tgt[:-1] if tgt is not None else self.generator
        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank, enc_state if dec_state is None else dec_state, memory_lengths=lengths)
        slen, batch_size, embedding_dim = decoder_outputs.shape
        logit = self.generator(decoder_outputs.view(-1, embedding_dim)).view(slen, batch_size, -1)
        return logit, decoder_outputs, attns, dec_state


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.rnn = stacked_rnn.StackedLSTM(1, input_size, input_size, 0.0)
        self.projection = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.LogSoftmax(dim=-1))


    def forward(self, input):
        dim = input.shape[-1]
        batch_size = input.shape[1]
        hidden = [[nu.zeros(batch_size, dim)], [nu.zeros(batch_size, dim)]]
        for emb_t in input.split(1):
            emb_t = emb_t.squeeze(0)#[batch, dim]
            _, hidden = self.rnn(emb_t, hidden)
        logit = self.projection(hidden[0][0])
        return logit


def make_loss_compute(vocab_size):
    weight = torch.ones(vocab_size)
    weight[data.NULL_ID] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    return criterion


def build_model(opt, vocab_size):
    embeddings = nn.Embedding(vocab_size, opt.word_vec_size, padding_idx=data.NULL_ID)
    if opt.position_encoding:
        pe = modules.PositionalEncoding(opt.dropout, opt.word_vec_size)
        embeddings = nn.Sequential(embeddings, pe)
    else:
        embeddings = nn.Sequential(embeddings)
    if opt.model_type == 'transformer':
        encoder = TransformerEncoder(embeddings, opt.transformer_enc_layers, opt.head_count, opt.transformer_hidden_size, opt.dropout)
        decoder = TransformerDecoder(encoder.embeddings, opt.transformer_dec_layers, opt.head_count, opt.transformer_hidden_size, opt.dropout)
    elif opt.model_type == 'rnn':
        encoder = RNNEncoder(opt.num_layers, vocab_size, opt.word_vec_size, opt.rnn_size, opt.bidirectional_encoder, opt.dropout)
        decoder = InputFeedRNNDecoder(encoder.embeddings, opt.num_layers, opt.bidirectional_encoder, opt.rnn_size, opt.attn_type, opt.dropout)
    model = NMTModel(encoder, decoder)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def build_discriminator(opt):
    model = Discriminator(opt.rnn_size)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


if __name__ == '__main__':
    encoder = RNNEncoder(2, 10000, 512, 512, True, 0.3, True)
