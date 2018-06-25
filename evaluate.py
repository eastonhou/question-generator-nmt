
import config
import models
import options
import data
import argparse

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


