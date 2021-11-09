from matplotlib import pyplot as plt
import os
from util import *
from run_scripts import *
from mnznet import *

FIGSIZE = (5, 3.75)
BB2AC = (0.5, -0.2)

def plot_signals(pos, _score, _target, _minimizer, _best_density, title, plotrange=1000):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    _score = _score.squeeze().detach().cpu().numpy()
    _target = _target.squeeze().detach().cpu().numpy()
    _minimizer = _minimizer.squeeze().detach().cpu().numpy()
    ax1.plot(pos[:plotrange], _score[:plotrange])
    ax1.title.set_text('K-mer score')
    # Plot Target
    ax2.plot(pos[:plotrange], _target[:plotrange])
    ax2.title.set_text('Target pattern')
    # Plot Minimizer Locations
    ax3.plot(pos[:plotrange], _minimizer[:plotrange])
    ax3.title.set_text(f'Minimizer locations, D={_best_density:.3f}')
    fig.tight_layout()
    directory = os.path.dirname(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(title)
    plt.close()


def plot_visualize(exp='chrXC', k=8, w=13, prefix='initial_', prefix2='Initial '):
    dir = f'./results_set3/{exp}_k{k}_w{w}/'
    config = make_config(w, k, l=500 * (w + k), device=0, seq_path=f'{exp}.seq', save_path=dir)
    env = Env(config['env'], seqlist=f"{SEQ_DIR}{config['seq_path']}")
    prionet = MinimizerNet(env.l, env.k, env.w, env.vocab_size, d=config['hidden_dim']).to(device)
    tmplnet = PeriodicNet(period=2.0 * math.pi / env.w, n_waves=env.w).to(device)

    prionet.load_state_dict(torch.load(dir + f'{prefix}score_net.pth'))
    tmplnet.load_state_dict(torch.load(dir + f'{prefix}template_net.pth'))
    prionet.eval()
    tmplnet.eval()
    all_scr = []
    all_tmp = []
    all_pos = []
    all_mnz = []
    for i in range(env.n):
        seq = env.seqlist[i].unsqueeze(0).to(device)
        pos = torch.tensor(np.arange(env.l - env.k + 1)).view(1, 1, -1).to(device)
        scr = prionet(seq)
        tmp = tmplnet(pos)
        _, _, mnz = prionet.density(scr)
        all_scr.append(scr.squeeze().cpu().detach())
        all_pos.append(pos.squeeze().cpu().detach() + i * (env.l - env.k + 1))
        all_tmp.append(tmp.squeeze().cpu().detach())
        all_mnz.append(mnz.squeeze().cpu().detach())
        del seq, tmp, pos, mnz

    scr = torch.cat(all_scr).numpy()
    tmp = torch.cat(all_tmp).numpy()
    pos = torch.cat(all_pos).numpy()
    mnz = torch.cat(all_mnz).numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    ax1.plot(pos[500:1000], scr[500:1000])
    ax1.title.set_text(f'{prefix2}Priority scores')

    ax2.plot(pos[500:1000], tmp[500:1000])
    ax2.title.set_text(f'{prefix2}Template scores')

    ax3.plot(pos[500:1000], mnz[500:1000])
    ax3.title.set_text(f'{prefix2}Minimizer locations')

    plt.xlabel('$k$-mer positions')
    fig.tight_layout()
    plt.savefig(f'./plots/{prefix}visualize.png')

def plot_learning_curve(learning_curve, title):
    plt.figure()
    directory = os.path.dirname(title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.plot(learning_curve['pos'], learning_curve['density'])
    plt.xlabel('Iteration')
    plt.ylabel('Best density factor')
    plt.savefig(title)
    plt.close()


def plot_compare_k(exp='chr1'):
    w = 14
    k_vals = [6, 8, 10, 12, 14]
    plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)
    for k in k_vals:
        dir = f'./results/{exp}_k{k}_w{w}/'
        summary = torch.load(dir + 'summary.pth')
        pos = summary['pos']
        density = np.vstack(summary['density']).squeeze()
        x, y = [pos[0]], [density[0]]
        for i in range(len(pos)):
            if pos[i] > 600:
                break
            if (pos[i] + 1) % 20 == 0:
                x.append(pos[i])
                y.append(density[i])
        ax.plot(x, y, '-o', label=f'k={k}', markevery=5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best Density Factor')
    ax.set_title(f'DeepMinimizer performance on {exp}')
    ax.legend(loc='upper center', bbox_to_anchor=BB2AC,
              fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./plots/mznet_{exp}_w{w}.png')


def plot_compare_w(exp='chr1'):
    k = 13
    w_vals = [10, 25, 40, 55, 70, 85]
    plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)
    for w in w_vals:
        dir = f'./results/{exp}_k{k}_w{w}/'
        summary = torch.load(dir + 'summary.pth')
        pos = summary['pos']
        density = np.vstack(summary['density']).squeeze()
        x, y = [pos[0]], [density[0]]
        for i in range(len(pos)):
            if pos[i] > 600:
                break
            if (pos[i] + 1) % 20 == 0:
                x.append(pos[i])
                y.append(density[i])
        ax.plot(x, y, '-o', label=f'w={w}', markevery=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Best Density Factor')
    ax.set_title(f'DeepMinimizer performance on {exp}')
    ax.legend(loc='upper center', bbox_to_anchor=BB2AC,
              fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./plots/mznet_{exp}_k{k}.png')


def plot_vary_k(exp='chr1'):
    w = 14
    k_vals = [6, 8, 10, 12, 14]
    pasha, mini, rnd, mnznet= [], [], [], []
    polar, polar_k_vals = [], []
    for k in k_vals:
        dir = f'./results/{exp}_k{k}_w{w}/'
        comparison = torch.load(dir + 'comparison.pth')
        mnznet.append(comparison['mnznet'])
        pasha.append(comparison['pasha'])
        mini.append(comparison['miniception'])
        rnd.append(comparison['random'])
        if os.path.exists(dir + 'polar.pth'):
            polar_k_vals.append(k)
            polar.append(torch.load(dir + 'polar.pth'))

    plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)
    ax.plot(k_vals, mnznet, '-o', label='DeepMinimizer')
    ax.plot(k_vals, pasha, '-o', label='PASHA')
    ax.plot(k_vals, mini, '-o', label='Miniception')
    ax.plot(k_vals, rnd, '-o', label='Random Minimizer')
    ax.plot(polar_k_vals, polar, '-o', label='PolarSet')
    ax.set_xlabel('k-mer length')
    ax.set_ylabel('Density factor')
    ax.legend(loc='upper center', bbox_to_anchor=BB2AC,
              fancybox=True, shadow=True, ncol=3)
    ax.set_title(f'Density factor vs. k-mer length on {exp}')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./plots/compare_k_{exp}.png')


def plot_vary_w(exp='chr1'):
    k = 13
    w_vals = [10, 25, 40, 55, 70, 85]
    pasha, mini, rnd, mnznet = [], [], [], []
    polar, polar_w_vals = [], []
    for w in w_vals:
        dir = f'./results/{exp}_k{k}_w{w}/'
        comparison = torch.load(dir + 'comparison_w.pth')
        mnznet.append(comparison['mnznet'])
        pasha.append(comparison['pasha'])
        mini.append(comparison['miniception'])
        rnd.append(comparison['random'])
        if os.path.exists(dir + 'polar.pth'):
            polar_w_vals.append(w)
            polar.append(torch.load(dir + 'polar.pth'))

    plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)
    ax.plot(w_vals, mnznet, '-o', label='DeepMinimizer')
    ax.plot(w_vals, pasha, '-o', label='PASHA')
    ax.plot(w_vals, mini, '-o', label='Miniception')
    ax.plot(w_vals, rnd, '-o', label='Random Minimizer')
    ax.plot(polar_w_vals, polar, '-o', label='PolarSet')
    ax.get_position()
    ax.set_xlabel('Window length')
    ax.set_ylabel('Density factor')
    ax.legend(loc='upper center', bbox_to_anchor=BB2AC,
               fancybox=True, shadow=True, ncol=3)
    ax.set_title(f'Density factor vs. window length on {exp}')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'./plots/compare_w_{exp}.png')

def compare_div_func_vary_k(exp='chr1'):
    w = 14
    k_vals = np.array([6, 8, 10, 12, 14])
    delta, l2 = [], []
    for k in k_vals:
        dir = f'./results_set2/{exp}_k{k}_w{w}/'
        comparison = torch.load(dir + 'comparison.pth')
        delta.append(comparison['maxpool_delta'])
        l2.append(comparison['l2'])
    delta = np.array(delta)
    l2 = np.array(l2)

    if exp == 'chr1': exp = 'hg38_all'
    plt.figure(figsize=FIGSIZE)
    ax = plt.subplot(111)
    ax.bar(k_vals - 0.2, delta, 0.4, label='$\Delta$')
    ax.bar(k_vals + 0.2, l2, 0.4, label='$\ell^2$')
    ax.set_xlabel("k-mer length")
    ax.set_ylabel("Density Factor")
    ax.legend(loc='upper center', bbox_to_anchor=BB2AC,
              fancybox=True, shadow=True, ncol=2, fontsize=15)
    ax.set_title(f'Density factor vs. k-mer length on {exp}')
    plt.tight_layout()
    plt.savefig(f'./plots/compare_divfunc_{exp}.png')

def load_polar_set_results(dat_file):
    f = open(dat_file)
    lines = f.readlines()
    wc = [26, 56, 86]
    for l in lines:
        tokens = l.split(',')
        seq, w, k, density = tokens[1], int(tokens[2]), tokens[3], float(tokens[4])
        if w in wc:
            w -= 1
        path = f'./results/{seq}_k{k}_w{w}'
        if os.path.exists(path):
            torch.save(density * (w + 1), f'{path}/polar.pth')


if __name__ == '__main__':
    plot_visualize(prefix='initial_', prefix2='Initial ')
    # plot_visualize(prefix='', prefix2='Final ')
    # load_polar_set_results('./master_dat_2')
    # load_polar_set_results('./master_dat')
    # plot_vary_k('chr1')
    # plot_vary_k('chrX')
    # plot_vary_k('chrXC')
    # plot_vary_k('hg38_all')
    # plot_vary_w('chr1')
    # plot_vary_w('chrX')
    # plot_vary_w('chrXC')
    # plot_vary_w('hg38_all')
    # plot_compare_k('chr1')
    # plot_compare_k('chrX')
    # plot_compare_k('chrXC')
    # plot_compare_k('hg38_all')
    # plot_compare_w('chr1')
    # plot_compare_w('chrX')
    # plot_compare_w('chrXC')
    # plot_compare_w('hg38_all')
    # compare_div_func_vary_k('chr1')
    # compare_div_func_vary_k('chrX')
    # compare_div_func_vary_k('chrXC')