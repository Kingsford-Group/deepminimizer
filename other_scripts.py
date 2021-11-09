from mnznet import *


def export_artifacts(env, scorenet, config, long_seq=True):
    save_folder = config['save_folder']
    with torch.no_grad():
        # Export sequence
        str = list(torch.argmax(env.seqlist, dim=1).cpu().numpy())
        f = open(f'./{save_folder}/seqs.txt', 'w')
        for i in range(env.n):
            str[i] = ''.join([env.vocab[str[i][j]] for j in range(env.l)])
            f.write(str[i] + '\n')
        f.close()

        # Export ordering (long seq)
        if long_seq:
            all_kmer = [''.join(p) for p in itertools.product(env.vocab, repeat=env.k)]
            all_kmer_onehot = [torch.tensor(np.array([env.c2i[c] for c in s])) for s in all_kmer]
            all_kmer_onehot = F.one_hot(torch.stack(all_kmer_onehot)).transpose(1, 2).float()
            n_kmer = all_kmer_onehot.shape[0]
            batch_size = 2000
            score = []
            counter = trange(0, n_kmer, batch_size)
            for i in counter:
                j = min(i + batch_size, n_kmer)
                batch = all_kmer_onehot[i: j].to(device)
                batch_score = scorenet(batch)
                score.append(batch_score)
                del batch
                torch.cuda.empty_cache()
                counter.set_postfix_str(cuda_memory(config['device']))
            score = torch.cat(score).cpu().detach().numpy()
            kmer_dict = dict(zip(all_kmer, score))
        # Export ordering (short seq)
        else:
            score = scorenet(env.seqlist)
            kmer_dict = {}
            for i in trange(env.n):
                for j in range(score.shape[-1]):
                    kmer = str[i][j: j + env.k]
                    if kmer in kmer_dict:
                        if score[i, 0, j] != kmer_dict[kmer]:
                            print(kmer, score[i, 0, j], kmer_dict[kmer])
                    kmer_dict[kmer] = score[i, 0, j]

        ordering = {k: v for k, v in sorted(kmer_dict.items(), key=lambda item: item[1])}
        f = open(f'./{save_folder}/order.txt', 'w')
        for k in ordering.keys():
            f.write(f'{k}, {ordering[k]}\n')
        f.close()



def plot_results():
    w_values = [15, 20, 25, 30, 35, 40, 45, 50]
    k_values = [5, 6, 7, 8, 9, 10]
    k_values_2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    plt.figure()
    for w in w_values:
        dw = []
        lc = torch.load(f'mnznet_w{w}/summary.pth')
        lc2 = torch.load(f'mnznet_w{w}/summary_11_25.pth')
        for k in k_values:
            avg_density = lc[k]['density'][-1]
            dw.append(avg_density)
        for k in k_values_2:
            avg_density = lc2[k]['density'][-1]
            dw.append(avg_density)
        plt.plot(k_values + k_values_2, dw, label=f'w={w}')
    plt.legend()
    plt.xticks(k_values + k_values_2)
    plt.savefig('./summary.png')
    plt.close()

    w = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    d = [1.758, 1.737, 1.712, 1.703, 1.693, 1.697, 1.699, 1.699, 1.693, 1.694, 1.694, 1.702, 1.701, 1.704, 1.705, 1.714,
         1.718, 1.730, 1.737, 1.742]
    plt.figure()
    plt.plot(w, d)
    plt.xticks([10, 15, 20, 25, 30])
    plt.savefig('./vary_w.png')
    plt.close()

def script_1():
    # Compute density for
    config = make_config(14, 6, l=10000, device=4, seq_path='chrX.seq', save_path='chrx')
    torch.cuda.set_device(config['device'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    batch_size = config['batch_size']
    env_path = f"./{config['save_folder']}/env.pt"
    # Creating Env and Load Networks
    env = Env(config['env'], seqlist=f"./{config['seq_path']}")
    net = MinimizerNet(env.l, env.k, env.w, env.vocab_size).to(device)
    net.load_state_dict(torch.load(f"./{config['save_folder']}/score_net.pth"))
    net.eval()
    x = []
    y = []
    with torch.no_grad():
        density = []
        batch_id = range(0, env.n, batch_size)
        for b, i in enumerate(batch_id):
            batch = env.seqlist[i: min(i + batch_size, env.n)].to(device)
            score = net(batch)
            batch_density, ave_batch_density, minimizer = net.density(score)
            density.append(batch_density)
            del batch, score, minimizer
            torch.cuda.empty_cache()
            x.append(i)
            y.append(ave_batch_density)

    plt.figure()
    plt.plot(x, y, '-')
    plt.savefig(f"./{config['save_folder']}/density_position.png")