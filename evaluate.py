
import config
import models
import options
import data
import argparse
import utils
import os
import torch
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
    feeder = data.TrainFeeder(dataset)
    model = models.build_model(opt, dataset.vocab_size)
    translator = Translator(model, opt.beam_size, opt.max_length)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
    feeder.prepare('dev')
    lines = []
    while not feeder.eof():
        pids, qids, _ = feeder.next(opt.batch_size)
        src = nu.tensor(pids)
        lengths = (src != data.NULL_ID).sum(-1)
        tgt = translator.translate(src.transpose(0, 1), lengths, opt.best_k_questions)
        passages = [feeder.ids_to_sent(t) for t in pids]
        questions = [[feeder.ids_to_sent(t) for t in qs] for qs in tgt]
        gtruths = [feeder.ids_to_sent(t) for t in qids]
        for p, qs, g in zip(passages, questions, gtruths):
            lines.append('--------------------------------')
            lines.append(p)
            lines.append('reference: ' + g)
            for k, q in enumerate(qs):
                lines.append('predict {}: {}'.format(k, q))
    utils.write_all_lines(opt.output_file, lines)


if __name__ == '__main__':
    evaluate()

