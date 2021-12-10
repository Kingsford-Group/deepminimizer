from util import *


class Env:
    # seq data: N * L * vocab_size
    def __init__(self, env_arg, seqlist=None):
        self.w = env_arg['window_size']
        self.k = env_arg['kmer_size']
        self.l = env_arg['sequence_length']
        self.n = env_arg['num_sequence']
        self.w_size = self.w + self.k - 1
        self.n_window = self.l - self.w_size
        self.vocab = env_arg['sequence_vocab']
        self.vocab_size = len(self.vocab)
        self.c2i = {self.vocab[i]: i for i in range(self.vocab_size)}
        self.vocab_prob = env_arg['sequence_vocab_probability']
        self.seqlist = None

        if seqlist is None:
            self.seqlist = self.generate_seqlist()
        else:
            self.seqlist = self.load_seqlist(seqlist)

    def generate_seqlist(self):
        dist = OneHotCategorical(probs=self.vocab_prob)
        seqlist = dist.sample((self.n, self.l)).transpose(1, 2)
        return seqlist

    def load_seqlist(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        seq = ''.join([line.strip() for line in lines])
        self.n = int(len(seq) / self.l) - 1
        seqs = [seq[i: i + self.l] for i in range(0, len(seq), self.l)]
        counter = trange(self.n)
        counter.set_description('Loading Seq Data')
        seqlist = [torch.tensor(np.array([self.c2i[c] for c in seqs[i]])) for i in counter]
        seqlist = F.one_hot(torch.stack(seqlist)).transpose(1, 2).float()
        del lines, seqs
        return seqlist

    def sample(self, n_sample=10):
        indices = torch.randperm(self.n)[:n_sample]
        return self.seqlist[indices].to(device)

    def save_env(self, folder):
        torch.save(self.__dict__, f'./{folder}/env.pt')

    def load_env(self, folder):
        self.__dict__ = torch.load(f'./{folder}/env.pt')
