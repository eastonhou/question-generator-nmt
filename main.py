import options
import models
import data
import config
import os
import json
import torch
import argparse
import nmt.utils as nu
from translator import Translator
from flask import Flask, jsonify
from urllib.parse import unquote
app = Flask(__name__)

def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.evaluate_opts(parser)
    return parser.parse_args()


def build_translator():
    opt = make_options()
    dataset = data.Dataset()
    feeder = data.Feeder(dataset)
    model = models.build_model(opt, dataset.vocab_size)
    translator = Translator(model, opt.beam_size, opt.max_length)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=lambda storage, location: storage)
        model.load_state_dict(ckpt['model'])
    return translator, feeder


def unique(sequences):
    seen = set()
    return [x for x in sequences if not (x in seen or seen.add(x))]


ckpt_path = os.path.join(config.checkpoint_folder, 'model.ckpt')
translator, feeder = build_translator()


@app.route('/gq/<document>', methods=['GET'])
def generate_question(document):
    document = unquote(document)
    pids = [feeder.sent_to_ids(document)]
    src = nu.tensor(pids)
    lengths = (src != data.NULL_ID).sum(-1)
    tgt = translator.translate(src.transpose(0, 1), lengths, 3)
    questions = [feeder.ids_to_sent(t) for t in tgt[0]]
    questions = unique(questions)
    obj = {
        'document': document,
        'questions': questions
    }
    return json.dumps(obj, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run()