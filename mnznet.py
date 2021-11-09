import torch.distributions

from util import *
from plot_scripts import *
from kmer_env import Env


class MinimizerNet(nn.Module):
    def __init__(self, l, k, w, vocab_size, d=256):
        super().__init__()
        self.l = l
        self.k = k
        self.w = w
        self.d = d
        self.vocab_size = vocab_size
        self.score_net = nn.Sequential(
            # kmer embedding 1: [N * V * L] -> [N * D * (L - K + 1)]
            nn.Conv1d(
                in_channels=vocab_size,
                out_channels=d,
                kernel_size=self.k
            ),
            nn.ReLU(),
            # kmer embedding 2: [N * D * (L - K + 1)] -> [N * (L - k + 1) * (D / 2)]
            nn.Conv1d(
                in_channels=d,
                out_channels=d//2,
                kernel_size=1
            ),
            nn.ReLU(),
            # kmer score: [N * D/2 * (L - K + 1)] -> [N * 1 * (L - k + 1)]
            nn.Conv1d(
                in_channels=d//2,
                out_channels=1,
                kernel_size=1
            ),
            nn.Sigmoid(),
        )
        self.maxpool = nn.MaxPool1d(w, stride=1, return_indices=True)
        self.unpool = nn.MaxUnpool1d(w, stride=1)

    # S dim: [N * V * L]
    def forward(self, S):
        # score dim: [N * 1 * (L * K - 1)]
        return self.score_net(S)

    def density(self, score, eps=1.0):
        val, idx = self.maxpool(score + eps)
        loc = self.unpool(val, idx, output_size=score.shape)
        minimizer = loc / ((loc == 0) + loc)
        density = (torch.sum(minimizer, dim=-1) / score.shape[-1]) * (self.w + 1)
        avg_density = torch.mean(density).detach().cpu().numpy()
        del val, idx, loc
        return density, avg_density, minimizer


class CosLayer(nn.Module):
    def __init__(self, amplitude=1.0, period=math.pi, shift=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(amplitude), requires_grad=True)
        self.p = nn.Parameter(torch.tensor(period), requires_grad=False)
        self.d = nn.Parameter(torch.tensor(shift), requires_grad=False)

    def forward(self, x):
        return self.a * torch.cos(self.p * (x + self.d))


class SeqConvNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=d, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=d, out_channels=d//2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=d//2, out_channels=c_out, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class PeriodicNet(nn.Module):
    def __init__(self, amplitude=0.5, period=math.pi, n_waves=3):
        super().__init__()
        self.n_waves = n_waves
        self.waves = nn.ModuleList([
            CosLayer(amplitude, period, shift=-1.0 * i) for i in range(n_waves)
        ])
        self.w = nn.Parameter(torch.rand(1, n_waves, 1), requires_grad=True)
        # self.net = SeqConvNet(c_in=n_waves, d=8)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        wave_signals = torch.cat([self.waves[i](x) for i in range(self.n_waves)], dim=1)
        wave_signals = wave_signals * self.sm(self.w)
        wave_signals = torch.sum(wave_signals, dim=1, keepdim=True)
        return torch.sigmoid(wave_signals)


def deep_minimizer(config):
    ##############################################
    # Initialize System
    torch.cuda.set_device(config['device'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    batch_size = config['batch_size']
    env_path = f"./{config['save_folder']}/env.pt"
    # Creating Env and Networks
    if os.path.exists(env_path):
        env = Env(config['env'])
        print('Loading Env')
        env.load_env(config['save_folder'])
    else:
        if not os.path.exists(os.path.dirname(env_path)):
            os.makedirs(os.path.dirname(env_path))
        if config['seq_path'] is None:
            env = Env(config['env'])
        else:
            env = Env(config['env'], seqlist=f"{SEQ_DIR}{config['seq_path']}")

    print(f'Env: N={env.n}, L={env.l}, W={env.w}, K={env.k}')
    mnznet = MinimizerNet(env.l, env.k, env.w, env.vocab_size, d=config['hidden_dim']).to(device)
    sinenet = PeriodicNet(period=2.0 * math.pi / env.w, n_waves=env.w).to(device)
    sn_title = f"./{config['save_folder']}/initial_score_net.pth"
    tn_title = f"./{config['save_folder']}/initial_template_net.pth"
    directory = os.path.dirname(sn_title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(mnznet.state_dict(), sn_title)
    torch.save(sinenet.state_dict(), tn_title)
    best_mnznet = deepcopy(mnznet)
    ##############################################

    # Eval function for entire sequence
    def _eval(net, ctr=None):
        net.eval()
        with torch.no_grad():
            density = []
            batch_id = range(0, env.n, batch_size)
            if ctr is None:
                eval_counter = tqdm(batch_id, desc='Evaluating Minimizer Net', leave=False)

            for b, i in enumerate(batch_id):
                if ctr is None: eval_counter.update(1)
                batch = env.seqlist[i: min(i + batch_size, env.n)].to(device)
                score = net(batch)
                batch_density, ave_batch_density, minimizer = net.density(score)
                density.append(batch_density)
                del batch, score, minimizer
                torch.cuda.empty_cache()
                if ctr is None:
                    eval_counter.set_postfix_str(
                        f'Batch density={ave_batch_density:.3f}'
                    )
                else:
                    ctr.set_postfix_str(
                        f'(Eval) Batch {b}/{len(batch_id)} '
                        f'density={ave_batch_density:.3f}'
                    )
            if ctr is None:
                eval_counter.close()
            return torch.mean(torch.cat(density)).detach().cpu().numpy()

    ##############################################
    # Optimizing Network
    opt = torch.optim.Adam([
        {'params': mnznet.parameters(), 'lr': config['mnznet_lr']},
        {'params': sinenet.parameters(), 'lr': config['sinenet_lr']}
    ])
    pos = torch.tensor(np.arange(env.l - env.k + 1)).view(1, 1, -1).to(device)
    # posnp = pos.squeeze().cpu().numpy()
    best_density = _eval(best_mnznet)
    learning_curve = {'pos': [0], 'density': [best_density]}
    for j in range(config['epoch']):
        batch = env.sample(batch_size)
        counter = trange(config['max_iter'], desc=f"Epoch {j}/{config['epoch']}", leave=False)
        for i in counter:
            mnznet.train()
            sinenet.train()
            opt.zero_grad()
            # Compute score & target
            score = mnznet(batch)
            target = sinenet(pos).repeat(batch_size, 1, 1)
            loss, lc = config['div_func'](score, target)
            loss.backward()
            opt.step()
            counter.set_postfix_str(
                f'I.{i}: L={lc[0]:.3f}, R1={lc[1]:.3f}, R2={lc[2]:.3f}, Best = {best_density:.3f}'
            )
            del target, score
            torch.cuda.empty_cache()
        # Record Best Performance
        if (j + 1) % config['eval_interval'] == 0:
            avg_density = _eval(mnznet, ctr=None)
            if avg_density < best_density:
                best_density = avg_density
                best_mnznet = deepcopy(mnznet)
                best_tplnet = deepcopy(sinenet)

                sn_title = f"./{config['save_folder']}/score_net.pth"
                tn_title = f"./{config['save_folder']}/template_net.pth"
                torch.save(best_mnznet.state_dict(), sn_title)
                torch.save(best_tplnet.state_dict(), tn_title)

            learning_curve['pos'].append(j)
            learning_curve['density'].append(best_density)
            plot_learning_curve(learning_curve, title=f"./{config['save_folder']}/learning_curve.png")
        del batch
        torch.cuda.empty_cache()

    return env, best_mnznet, learning_curve, best_density




