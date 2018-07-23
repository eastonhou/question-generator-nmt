import argparse
import options
import models
import torch
import config
import data
import utils
import os
import random
import nmt.utils as nu
import evaluate

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
    model.train()
    nbatch = 0
    vocab_size = feeder.dataset.vocab_size
    criterion = models.make_loss_compute(vocab_size)
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
    generator.train()
    discriminator.train()
    nbatch = 0
    vocab_size = feeder.dataset.vocab_size
    g_criterion =  models.make_loss_compute(feeder.dataset.vocab_size)
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
            d_loss = d_criterion(d_logit, flag)
            loss = g_loss + d_loss
            print('------{} {}, {}/{}, loss: {:>.4F}+{:>.4F}={:>.4F}'.format(step, feeder.iteration, feeder.cursor, feeder.size, g_loss.tolist(), d_loss.tolist(), loss.tolist()))
        else:
            tc_logit = discriminator(tc_hidden)
            fr_logit = discriminator(fr_hidden)
            logit = torch.cat([tc_logit, fr_logit], dim=0)
            flag = nu.tensor([1]*batch_size + [0]*batch_size)
            loss = d_criterion(logit, flag)
            print('------{} {}, {}/{}, loss: {:>.4F}'.format(step, feeder.iteration, feeder.cursor, feeder.size, loss.tolist()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if nbatch % 10 == 0:
            logit = z.transpose(0, 1)
            gids = logit.argmax(-1).tolist()
            for k in range(len(gids)):
                question = feeder.ids_to_sent(gids[k])
                print('truth:   {}'.format(feeder.ids_to_sent(qids[k])))
                print('predict: {}'.format(question))
                print('----------')
    return loss


def train(auto_stop, steps=200, evaluate_size=500):
    opt = make_options()
    generator, discriminator, g_optimizer, d_optimizer, feeder, ckpt = models.load_or_create_models(opt, True)
    if ckpt is not None:
        last_accuracy = evaluate.evaluate_accuracy(generator, feeder.dataset, size=evaluate_size)
    else:
        last_accuracy = 0
    while True:
        if opt.using_gan == 1:
            mini_steps = steps // 10
            state = feeder.state()
            run_gan_epoch(opt, generator, discriminator, feeder, d_optimizer, mini_steps, 'discriminator')
            feeder.load_state(state)
            run_gan_epoch(opt, generator, discriminator, feeder, g_optimizer, mini_steps*9, 'generator')
        else:
            run_epoch(opt, generator, feeder, g_optimizer, steps)
        accuracy = evaluate.evaluate_accuracy(generator, feeder.dataset, size=evaluate_size)
        if accuracy > last_accuracy:
            utils.mkdir(config.checkpoint_folder)
            models.save_models(opt, generator, discriminator, g_optimizer, d_optimizer, feeder)
            last_accuracy = accuracy
            print('MODEL SAVED WITH ACCURACY {:>.2F}.'.format(accuracy))
        else:
            if random.randint(0, 4) == 0:
                models.restore(generator, discriminator, g_optimizer, d_optimizer)
                print('MODEL RESTORED {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))
            else:
                print('CONTINUE TRAINING {:>.2F}/{:>.2F}.'.format(accuracy, last_accuracy))

train(False)