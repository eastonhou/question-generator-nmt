import config
import utils
import random
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
        self.vocab_size = len(self.c2i)
        self.train_set = load_qa(config.train_file, config.answer_limit)
        self.dev_set = load_qa(config.dev_file, config.answer_limit)


class Feeder(object):
    def __init__(self, dataset):
        self.dataset = dataset


    def stoi(self, word):
        if word in self.dataset.c2i:
            return self.dataset.c2i[word]
        else:
            return OOV_ID


    def sent_to_ids(self, sent):
        return [self.stoi(w) for w in sent]


    def ids_to_sent(self, ids):
        return ''.join([self.dataset.i2c[id] for id in ids if id != NULL_ID])


    def decode_logit(self, logit):
        ids = np.argmax(logit, -1)
        sent = self.ids_to_sent(ids)
        return sent


class TrainFeeder(Feeder):
    def __init__(self, dataset):
        super(TrainFeeder, self).__init__(dataset)


    def prepare(self, type):
        if type == 'train':
            self.prepare_data(self.dataset.train_set)
            random.shuffle(self.data_index)
        elif type == 'dev':
            self.prepare_data(self.dataset.dev_set)
            self.keep_prob = 1.0
        self.cursor = 0
        self.iteration = 1
        self.size = len(self.data)


    def prepare_data(self, dataset):
        self.data = dataset
        self.data_index = list(range(len(self.data)))
        r = set()
        for _, questions in self.data:
            for question in questions:
                r.add(tuple(question))
        self.questions = [list(q) for q in r]


    def state(self):
        return self.iteration, self.cursor, self.data_index


    def load_state(self, state):
        self.iteration, self.cursor, self.data_index = state


    def nonstopwords(self, sent):
        return set(sent) - self.dataset.stopwords


    def eof(self):
        return self.cursor == self.size


    def next(self, batch_size, align=True):
        if self.eof():
            self.iteration += 1
            self.cursor = 0
        size = min(self.size - self.cursor, batch_size)
        batch = self.data_index[self.cursor:self.cursor+size]
        batch = [self.data[idx] for idx in batch]
        batch_pid = []
        batch_qid = []
        batch_target = []
        batch = sorted(batch, key=lambda x:-len(x[0]))
        for example in batch:
            passage, question = example
            passage, question = passage, question[0]
            target = SOS + question + EOS
            pids = self.sent_to_ids(passage)
            qids = self.sent_to_ids(question)
            tids = self.sent_to_ids(target)
            batch_pid.append(pids)
            batch_qid.append(qids)
            batch_target.append(tids)
        self.cursor += size
        return align2d(batch_pid) if align else batch_pid, align2d(batch_qid) if align else batch_qid, align2d(batch_target) if align else batch_target


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


def load_qa(filename, answer_limit):
    lines = []
    r = []
    for line in utils.read_all_lines(filename):
        if line == '<P>':
            passage = lines[0].replace(' ', '')
            if len(passage) <= answer_limit:
                questions = [q.replace(' ', '') for q in lines[1:]]
                if questions:
                    r.append((passage, questions))
            lines.clear()
        else:
            lines.append(line)
    return r


def align2d(values, fill=0):
    mlen = max([len(row) for row in values])
    return [row + [fill] * (mlen - len(row)) for row in values]


def align3d(values, fill=0):
    lengths = [[len(x) for x in y] for y in values]
    maxlen0 = max([max(x) for x in lengths])
    maxlen1 = max([len(x) for x in lengths])
    for row in values:
        for line in row:
            line += [fill] * (maxlen0 - len(line))
        row += [([fill] * maxlen0)] * (maxlen1 - len(row))
    return values