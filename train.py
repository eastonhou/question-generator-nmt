import argparse
import options
import models
import torch
import data

def make_options():
    parser = argparse.ArgumentParser(description='train.py')
    options.model_opts(parser)
    options.train_opts(parser)
    return parser


def make_loss_compute(generator, vocab):
    weight = torch.ones(len(vocab.itos))
    weight[data.NULL_ID] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    return criterion


def build_model(opt, vocab):
    vocab_size = len(vocab.itos)
    encoder = models.RNNEncoder(opt.num_layers, vocab_size, opt.word_vec_size, opt.rnn_size, opt.bidirectional_encoder, opt.dropout)
    decoder = models.InputFeedRNNDecoder(encoder.embeddings, opt.num_layers, opt.bidirectional_encoder, opt.rnn_size, opt.attn_type, opt.dropout)
    model = models.NMTModel(encoder, decoder)
    if opt.use_gpu:
        model = model.cuda()
    criterion = make_loss_compute(encoder.embeddings.weight, vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    return model, criterion, optimizer



    
