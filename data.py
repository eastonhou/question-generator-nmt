import config
import utils
import numpy as np

NULL = '<NULL>'
OOV = '<OOV>'
SOS = '<S>'
EOS = '</S>'

NULL_ID = 0
OOV_ID = 1
SOS_ID = 2
EOS_ID = 3


class Dataset(object):
    def __init__(self):
        self.c2i, self.i2c, self.i2n = load_vocab(config.vocab_file, config.vocab_size)
        self.chars = list(self.c2i.keys())
        self.char_weights = [self.i2n[id] for id in range(len(self.chars))]
        self.norm_char_weights = self.char_weights / np.sum(self.char_weights)
#        self.train_set = load_qa(config.train_file, config.answer_limit)
#        self.dev_set = load_qa(config.dev_file, config.answer_limit)


def load_vocab(filename, count):
    w2i = {
        NULL: NULL_ID,
        OOV: OOV_ID,
        SOS: SOS_ID,
        EOS: EOS_ID
    }
    i2c = {
        NULL_ID: 0,
        SOS_ID: 0,
        EOS_ID: 0
    }
    all_entries = list(utils.read_all_lines(filename))
    count -= len(w2i)
    count = min(count, len(all_entries))
    for line in all_entries[:count]:
        word, freq = line.rsplit(':', 1)
        id = len(w2i)
        w2i[word] = id
        i2c[id] = int(freq)
    i2w = {k:v for v,k in w2i.items()}
    i2c[OOV_ID] = len(all_entries) - count
    return w2i, i2w, i2c