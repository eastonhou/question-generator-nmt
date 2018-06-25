import argparse
import options
import models
import torch
import config
import data
import utils
import os
import nmt.utils as nu

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.train_opts(parser)
    return parser.parse_args()


def make_loss_compute(vocab_size):
    weight = torch.ones(vocab_size)
    weight[data.NULL_ID] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    return criterion


def build_model(opt, vocab_size):
    encoder = models.RNNEncoder(opt.num_layers, vocab_size, opt.word_vec_size, opt.rnn_size, opt.bidirectional_encoder, opt.dropout)
    decoder = models.InputFeedRNNDecoder(encoder.embeddings, opt.num_layers, opt.bidirectional_encoder, opt.rnn_size, opt.attn_type, opt.dropout)
    model = models.NMTModel(encoder, decoder)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = make_loss_compute(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    return model, criterion, optimizer


def print_prediction(feeder, similarity, pids, qids, labels, number=None):
    if number is None:
        number = len(pids)
    for k in range(min(len(pids), number)):
        pid, qid, sim, lab = pids[k], qids[k], similarity[k], labels[k]
        passage = feeder.ids_to_sent(pid)
        print(passage)
        question = feeder.ids_to_sent(qid)
        print(' {} {:>.4F}: {}'.format(lab, sim, question))


def run_epoch(opt, model, feeder, criterion, optimizer, batches):
    nbatch = 0
    vocab_size = feeder.dataset.vocab_size
    while nbatch < batches:
        pids, qids, tids = feeder.next(opt.batch_size)
        batch_size = len(pids)
        nbatch += 1
        x = nu.tensor(pids)
        y = nu.tensor(qids)
        t = nu.tensor(tids)
        lengths = (x != data.NULL_ID).sum(-1)
        outputs, _, _ = model(x.transpose(0, 1), t.transpose(0, 1), lengths)
        loss = criterion(outputs.view(-1, vocab_size), t.transpose(0, 1)[1:].contiguous().view(-1)) / nu.tensor(batch_size).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(
            feeder.iteration, feeder.cursor, feeder.size, loss.tolist()))
        if nbatch % 10 == 0:
            logit = model(x, None)
            gids = logit.argmax(-1).tolist()
            for k in range(len(gids)):
                question = feeder.ids_to_sent(gids[k])
                print('truth:   {}'.format(feeder.ids_to_sent(qids[k][0])))
                print('predict: {}'.format(question))
                print('----------')
    return loss


def train(auto_stop, steps=50, threshold=0.2):
    opt = make_options()
    dataset = data.Dataset()
    feeder = data.TrainFeeder(dataset)
    model, criterion, optimizer = build_model(opt, dataset.vocab_size)
    feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        feeder.load_state(ckpt['feeder'])
    while True:
        #run_generator_epoch(generator, discriminator, generator_feeder, criterion, generator_optimizer, 0.2, 100)
        run_epoch(opt, model, feeder, criterion, optimizer, steps)
        utils.mkdir(config.checkpoint_folder)
        torch.save({
            'model':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'feeder': feeder.state()
            }, ckpt_path)
        print('MODEL SAVED.')


train(False)