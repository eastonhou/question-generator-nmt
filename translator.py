import argparse
import torch
import codecs
import os
import math
import data
import nmt.utils as nu
from itertools import count

class Beam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size 
        self.sid = nu.tensor(range(beam_size))
        self.cid = nu.tensor([data.NULL_ID if i != 0 else data.SOS_ID for i in range(beam_size)])
        self.seq = [list() for _ in range(beam_size)]
        self.scores = nu.tensor([0] * beam_size).float()


    def advance(self, scores):
        '''
        scores -> Tensor(beam_size, vocab_size)
        '''
        vocab_size = scores.shape[1]
        flat_scores = (scores + self.scores.unsqueeze(1).expand_as(scores)).view(-1)
        self.scores, ids = flat_scores.topk(self.beam_size)
        self.sid = ids / vocab_size
        self.cid = ids - self.sid * vocab_size
        self.seq = [self.seq[i].copy() for i in self.sid.tolist()]
        for seq, id in zip(self.seq, self.cid.tolist()):
            seq.append(id)


    def current_id(self):
        return self.cid


    def state_id(self):
        return self.state_id


    def sequence(self):
        return self.seq[0]


class Translator(object):

    def __init__(self, model, beam_size, max_length):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length


    def translate(self, src, src_lengths):
        batch_size = src_lengths.shape[0]
        enc_final, memory_bank = self.model.encoder(src, src_lengths)
        dec_states = self.model.decoder.init_decoder_state(src, memory_bank, enc_final)
        memory_bank = memory_bank.repeat(1, self.beam_size, 1)
        memory_lengths = src_lengths.repeat(self.beam_size)
        dec_states.repeat_beam_size_times(self.beam_size)
        beam = [Beam(self.beam_size) for _ in range(batch_size)]
        for _ in range(self.max_length):
            if all([b.done() for b in beam]):
                break
            inp = torch.stack([b.current_id() for b in beam]).t().view(1, batch_size*self.beam_size)#1 timestep
            dec_out, dec_states, _ = self.model.decoder(inp, memory_bank, dec_states, memory_lengths=memory_lengths)
            dec_out = dec_out.squeeze(0)#[beam_size*batch_size, rnn_size]
            out = self.model.generator(dec_out)#[beam_size*batch_size, vocab_size]
            out = out.view(self.beam_size, batch_size, -1)#[beam_size, batch_size, vocab_size]
            for k in len(beam):#logit~[beam_size, vocab_size]
                beam[k].advance(out[:,k,:])
                dec_states.beam_update(k, beam[k].state_id(), self.beam_size)
        return [b.sequence() for b in beam]

