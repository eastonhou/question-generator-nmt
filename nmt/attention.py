import torch
import torch.nn as nn
import nmt.utils as utils
import math

class GlobalAttention(nn.Module):
    def __init__(self, dim, coverage=False, attn_type='dot'):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        if self.attn_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        out_bias = self.attn_type == 'mlp'
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)
        self.sm = nn.Softmax(dim=-1)
        if coverage:
            self.inear_cover = nn.Linear(1, dim, bias=False)


    def score(self, ht, hs):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`
        """
        src_batch, src_len, _ = hs.shape
        tgt_batch, tgt_len, tgt_dim = ht.shape
        if self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                ht_ = ht.view(tgt_batch*tgt_len, tgt_dim)
                ht_ = self.linear_in(ht_)
                ht = ht_.view(tgt_batch, tgt_len, tgt_dim)
            return torch.bmm(ht, hs.transpose(1, 2))
        else:
            wq = self.linear_query(ht.view(-1, self.dim))
            wq = wq.view(tgt_batch, tgt_len, 1, self.dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, self.dim)
            uh = self.linear_context(hs.contiguous().view(-1, self.dim))
            uh = uh.view(src_batch, 1, src_len, self.dim)
            uh = uh.expand(src_batch, tgt_len, src_len, self.dim)
            wquh = (wq+uh).tanh()
            return self.v(wquh.view(-1, self.dim)).view(tgt_batch, tgt_len, src_len)


    def forward(self, input, memory_bank, memory_lengths):
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, src_len, dim = memory_bank.shape
        _, tgt_len, _ = input.shape

        align = self.score(input, memory_bank)

        mask = utils.sequence_mask(memory_lengths).unsqueeze(1)
        align.data.masked_fill_(1-mask, -float('inf'))

        align_vectors = self.sm(align.view(batch*tgt_len, src_len)).view(batch, tgt_len, src_len)
        c = torch.bmm(align_vectors, memory_bank)
        concat_c = torch.cat([c, input], 2).view(batch * tgt_len, self.dim*2)
        attn_h = self.linear_out(concat_c).view(batch, tgt_len, self.dim)
        if self.attn_type in ['general', 'dot']:
            attn_h = attn_h.tanh()

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, align_vectors


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout):
        assert model_dim % head_count == 0
        super(MultiHeadedAttention, self).__init__()
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, model_dim)
        self.linear_values = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)
        self.sm = nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)


    def forward(self, key, value, query, mask=None):
        '''
        args:
            key: [batch, key_len, dim]
            value: [batch, key_len, dim]
            query: [batch, query_len, dim]
            mask: [batch, query_len, key_len]
        returns:
            output: [batch, query_len, dim]
            attn: [batch, query_len, key_len]
        '''
        batch_size = key.shape[0]
        key_len = key.shape[1]
        query_len = query.shape[1]

        def shape(x):
            return x.view(batch_size, -1, self.head_count, self.dim_per_head).transpose(1, 2)#[batch, head_count, len, dim_per_head]

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)#[batch, len, dim]


        key_up = shape(self.linear_keys(key))#[batch, head_count, key_len, dim_per_head]
        value_up = shape(self.linear_values(value))#[batch, head_count, value_len, dim_per_head]
        query_up = shape(self.linear_query(query)) / math.sqrt(self.dim_per_head)#[batch, head_count, query_len, dim_per_head]
        scores = torch.matmul(query_up, key_up.transpose(2, 3))#[batch, head_count, query_len, key_len]
        mask = mask.unqueeze(1).expand_as(scores)#[batch, head_count, query_len, key_len]
        scores = scores.masked_fill(mask, -1E18)#[batch, head_count, query_len, key_len]
        attn = self.sm(scores)#[batch, head_count, query_len, key_len]
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value_up)#[batch, head_count, query_len, dim_per_head]
        context = unshape(context)#[batch, query_len, dim]
        output = self.final_linear(context)#[batch, query_len, dim]
        top_attn = attn.view(batch_size, self.head_count, query_len, key_len)[:, 0, :, :].contiguous()#[batch, query_len, key_len]
        return output, top_attn