import argparse



def model_opts(parser):
    group = parser.add_argument_group('embeddings')
    group.add_argument('-word_vec_size', type=int, default=512)
    group = parser.add_argument_group('rnn')
    group.add_argument('-rnn_size', type=int, default=512)


