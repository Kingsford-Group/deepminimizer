import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions import Categorical
from torch.optim import Adam

from copy import deepcopy
from collections import deque
from pprint import pprint
import subprocess
import sys
import os
import itertools
from queue import Queue
import time
import math
from tqdm import trange, tqdm
import random
from urllib import request
chmap = {'C': 0, 'G': 1, 'A': 2, 'T': 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_DIR = '/home/hongyuz/src/anchorsets/'

def state_dict_to_tensor(state_dict):
    return torch.cat([t.view(-1) for t in state_dict.values()])


def tensor_to_state_dict(w, state_dict):
    w = w.squeeze()
    curr = 0
    for k in state_dict:
        tsize = torch.numel(state_dict[k])
        state_dict[k] = w[curr: curr + tsize].reshape(state_dict[k].shape)
        curr += tsize


def ensure_not_1D(x):
    if x.ndim == 1:
        if isinstance(x, np.ndarray):
            x = torch.tensor(np.expand_dims(x, axis=0))
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(0)
    return x


def cuda_memory(device):
    t = torch.cuda.get_device_properties(device).total_memory
    c = torch.cuda.memory_cached(device)
    a = torch.cuda.memory_allocated(device)
    return f'Total:{t}, Cached:{c}, Alloc:{a}, Free:{t-c-a}'


def print_cuda_memory(device, prefix=''):
    print(prefix + cuda_memory(device))


def sequence_mer_iterator(k, seq):
    slen = len(seq)
    mod_low = 4 ** (k-1)
    cur = 0
    for i in range(k-1):
        cur = cur * 4 + chmap[seq[i]]
    for i in range(k-1, slen):
        if i >= k:
            cur -= mod_low * chmap[seq[i-k]]
        cur = cur * 4 + chmap[seq[i]]
        yield cur


def random_sequence(slen, seed = None):
    """
    Generates a random sequence.
    """
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice('ACTG') for _ in range(slen))


def kmer_to_int(km):
    '''
    Converts a k-mer to corresponding integer representation, following big-endian
    representation (leftmost character has highest weight).
    @param km: the k-mer.
    '''
    ret = 0
    for c in km:
        ret = ret * 4 + chmap[c]
    return ret
