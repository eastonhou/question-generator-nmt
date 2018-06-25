import argparse


def environment_opts(parser):
    group = parser.add_argument_group('environment')
    group.add_argument('-use_gpu', type=bool, default=False)


def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-word_vec_size', type=int, default=512)
    group.add_argument('-rnn_size', type=int, default=512)
    group.add_argument('-bidirectional_encoder', type=bool, default=True)
    group.add_argument('-attn_type', type=str, default='general')


def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-batch_size', type=int, default=64)
    group.add_argument('-learning_rate', type=float, default=0.001)
    group.add_argument('-dropout', type=float, default=0.3)




