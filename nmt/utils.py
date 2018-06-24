import torch


def use_gpu(opt):
    return hasattr(opt, 'gpu') and opt.gpu >= 0


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arrange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))
