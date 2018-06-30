
import config
import models
import options
import data
import argparse
import utils
import os
import torch
import random
import nmt.utils as nu
from translator import Translator

ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')

def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.evaluate_opts(parser)
    return parser.parse_args()


def evaluate():
    opt = make_options()
    dataset = data.Dataset()
    model = models.build_model(opt, dataset.vocab_size)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
    evaluate_accuracy(model, dataset, opt.batch_size, opt.beam_size, opt.min_length, opt.max_length, opt.best_k_questions, opt.output_file)


def evaluate_policy_docs():
    opt = make_options()
    dataset = data.Dataset()
    feeder = data.Feeder(dataset)
    model = models.build_model(opt, dataset.vocab_size)
    translator = Translator(model, opt.beam_size, opt.min_length, opt.max_length)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
    docs = data.load_policy_documents()
    for doc in docs:
        data.parse_paragraphs(doc)
    lines = []
    for doc in docs:
        paras = [p for p in doc.paragraphs if 50 <= len(p) <= 400]
        if not paras:
            continue
        lines.append('=================================')
        lines.append(doc.title)
        if len(paras) > 16:
            paras = random.sample(paras, 16)
        paras = sorted(paras, key=lambda x:-len(x))
        pids = [feeder.sent_to_ids(p) for p in paras]
        pids = data.align2d(pids)
        src = nu.tensor(pids)
        lengths = (src != data.NULL_ID).sum(-1)
        tgt = translator.translate(src.transpose(0, 1), lengths, opt.best_k_questions)
        questions = [[feeder.ids_to_sent(t) for t in qs] for qs in tgt]
        for p, qs in zip(paras, questions):
            lines.append('--------------------------------')
            lines.append(p)
            for k, q in enumerate(qs):
                lines.append('predict {}: {}'.format(k, q))
    utils.write_all_lines(opt.output_file, lines)
        

def evaluate_accuracy(model, dataset, batch_size=50, beam_size=5, min_length=5, max_length=20, best_k_questions=3, size=None, output_file=config.evaluate_output_file):
    feeder = data.TrainFeeder(dataset)
    feeder.prepare('dev')
    translator = Translator(model, beam_size, min_length, max_length)
    size = size or feeder.size
    lines = []
    correct = 0
    total = 0
    while feeder.cursor < size:
        x, _, lengths, pids, qids = data.next(feeder, batch_size)
        tgt = translator.translate(x, lengths, best_k_questions)
        passages = [feeder.ids_to_sent(t) for t in pids]
        questions = [[feeder.ids_to_sent(t) for t in qs] for qs in tgt]
        gtruths = [feeder.ids_to_sent(t) for t in qids]
        for p, qs, g in zip(passages, questions, gtruths):
            lines.append('--------------------------------')
            lines.append(p)
            lines.append('reference: ' + g)
            for k, q in enumerate(qs):
                lines.append('predict {}: {}'.format(k, q))
            correct += len(set(g) & set(qs[0]))
            total += len(set(qs[0]))
        print('{}/{}'.format(feeder.cursor, size))
    accuracy = correct/total*100
    lines.append('correct: {}/{}, accuracy: {}'.format(correct, total, accuracy))
    print('evauation finished with accuracy: {:>.2F}'.format(accuracy))
    utils.write_all_lines(output_file, lines)
    return accuracy


if __name__ == '__main__':
    evaluate()

