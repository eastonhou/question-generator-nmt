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


def print_prediction(feeder, similarity, pids, qids, labels, number=None):
    if number is None:
        number = len(pids)
    for k in range(min(len(pids), number)):
        pid, qid, sim, lab = pids[k], qids[k], similarity[k], labels[k]
        passage = feeder.ids_to_sent(pid)
        print(passage)
        question = feeder.ids_to_sent(qid)
        print(' {} {:>.4F}: {}'.format(lab, sim, question))


def run_epoch(opt, model, feeder, optimizer, batches):
    nbatch = 0
    vocab_size = feeder.dataset.vocab_size
    criterion = models.make_loss_compute(model.vocab_size)
    while nbatch < batches:
        x, t, lengths, _, qids = data.next(feeder, opt.batch_size)
        batch_size = lengths.shape[0]
        nbatch += 1
        outputs, _, _, _ = model(x, t, lengths)
        loss = criterion(outputs.view(-1, vocab_size), t[1:].contiguous().view(-1)) / nu.tensor(batch_size).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('------ITERATION {}, {}/{}, loss: {:>.4F}'.format(
            feeder.iteration, feeder.cursor, feeder.size, loss.tolist()))
        if nbatch % 10 == 0:
            logit = outputs.transpose(0, 1)
            gids = logit.argmax(-1).tolist()
            for k in range(len(gids)):
                question = feeder.ids_to_sent(gids[k])
                print('truth:   {}'.format(feeder.ids_to_sent(qids[k])))
                print('predict: {}'.format(question))
                print('----------')
    return loss


def run_gan_epoch(opt, generator, discriminator, feeder, optimizer, batches, step):
    nbatch = 0
    vocab_size = feeder.dataset.vocab_size
    g_criterion =  models.make_loss_compute(generator.vocab_size)
    d_criterion = torch.nn.NLLLoss()
    while nbatch < batches:
        x, t, lengths, _, qids = data.next(feeder, opt.batch_size)
        batch_size = lengths.shape[0]
        nbatch += 1
        y, tc_hidden, _, _ = generator(x, t, lengths)
        z, fr_hidden, _, _ = generator(x, None, lengths)
        if step == 'generator':
            g_loss = g_criterion(y.view(-1, vocab_size), t[1:].contiguous().view(-1)) / nu.tensor(batch_size).float()
            d_logit = discriminator(fr_hidden)
            flag = nu.tensor([1]*batch_size)
            loss = g_loss + d_criterion(d_logit, flag)
        else:
            tc_logit = discriminator(tc_hidden)
            fr_logit = discriminator(fr_hidden)
            logit = torch.cat([tc_logit, fr_logit], dim=1)
            flag = nu.tensor([1]*batch_size + [0]*batch_size)
            loss = d_criterion(logit, flag)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('------ITERATION[{}] {}, {}/{}, loss: {:>.4F}'.format(
            step, feeder.iteration, feeder.cursor, feeder.size, loss.tolist()))
        if nbatch % 10 == 0:
            logit = z.transpose(0, 1)
            gids = logit.argmax(-1).tolist()
            for k in range(len(gids)):
                question = feeder.ids_to_sent(gids[k])
                print('truth:   {}'.format(feeder.ids_to_sent(qids[k])))
                print('predict: {}'.format(question))
                print('----------')
    return loss


def train(auto_stop, steps=100):
    opt = make_options()
    dataset = data.Dataset()
    feeder = data.TrainFeeder(dataset)
    generator = models.build_model(opt, dataset.vocab_size)
    discriminator = models.build_discriminator(opt)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate)
    feeder.prepare('train')
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        generator.load_state_dict(ckpt['generator'])
        discriminator.load_state_dict(ckpt['discriminator'])
        g_optimizer.load_state_dict(ckpt['generator_optimizer'])
        d_optimizer.load_state_dict(ckpt['discriminator_optimizer'])
        feeder.load_state(ckpt['feeder'])
    while True:
        run_gan_epoch(opt, generator, discriminator, feeder, g_optimizer, steps, 'generator')
        run_gan_epoch(opt, generator, discriminator, feeder, d_optimizer, steps, 'discriminator')
        utils.mkdir(config.checkpoint_folder)
        torch.save({
            'generator':  generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'generator_optimizer': g_optimizer.state_dict(),
            'discriminator_optimizer': d_optimizer.state_dict(),
            'feeder': feeder.state()
            }, ckpt_path)
        print('MODEL SAVED.')


train(False)