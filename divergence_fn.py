from util import *


def l2_div(score, target):
    l1 = torch.norm(score - target)
    lt = torch.pow(torch.norm(target), 2.0)
    return l1 - lt, [l1, l1, lt]


def smooth_delta_div(score, target):
    l1 = torch.sum(target * torch.pow(target - score, 2.0))
    l2 = torch.sum((1.0 - target) * torch.pow(score, 2.0))
    lt = torch.pow(torch.norm(target), 2.0)
    return l1 + l2 - lt, [l1, l2, lt]


def maxpool_delta_div(score, target, w=13):
    mp = nn.MaxPool1d(w, stride=1, return_indices=True)
    up = nn.MaxUnpool1d(w, stride=1)
    peak, loc = mp(score)
    peak = up(peak, loc, output_size=score.shape)

    l1 = torch.sum(target * torch.pow(target - score, 2.0))
    l2 = torch.pow(torch.norm(score - peak), 2.0)
    lt = torch.pow(torch.norm(target), 2.0)
    return l1 + l2 - lt, [l1, l2, lt]
