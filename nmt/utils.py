import torch

def tensor(v):
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.tensor(v).cuda() if gpu_available() else torch.tensor(v)


def zeros(*kwargs):
    v = torch.zeros(kwargs)
    return v.cuda() if gpu_available() else v


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))


def gpu_available():
    return torch.cuda.is_available()