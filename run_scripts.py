from mnznet import *
from minimizer import *
from other_scripts import *
from divergence_fn import *


# def random_search(config):
#     # Initialize System
#     torch.cuda.set_device(config['device'])
#     np.random.seed(config['seed'])
#     torch.manual_seed(config['seed'])
#     env_arg = config['env']
#     folder = config['save_folder']
#     env_path = config['env_path']
#     max_iter = config['max_iter']
#     # Creating Env and Networks
#     env = Env(env_arg, seqlist=f'./{env_path}')
#     mnznet = MinimizerNet(env.l, env.k, env.w, env.vocab_size).to(device)
#     mnznet.eval()
#     state_dict = mnznet.state_dict()
#     weights = state_dict_to_tensor(state_dict)
#     best_mnznet = deepcopy(mnznet)
#     _, best_density, _ = best_mnznet.density(best_mnznet(env.seqlist))
#     learning_curve = {'pos': [0], 'density': [best_density]}
#     plot_learning_curve(learning_curve, title=f'./{folder}/learning_curve.png')
#     counter = trange(max_iter)
#     for i in counter:
#         weights = torch.randn(weights.shape)
#         tensor_to_state_dict(weights, state_dict)
#         mnznet.load_state_dict(state_dict)
#         _, avg_density, _ = mnznet.density(mnznet(env.seqlist))
#         if avg_density < best_density:
#             best_density = avg_density
#             best_mnznet = deepcopy(mnznet)
#             torch.save(best_mnznet, f'./{folder}/score_net.pth')
#         learning_curve['pos'].append(i)
#         learning_curve['density'].append(best_density)
#         if i % 2000 == 1999:
#             plot_learning_curve(learning_curve, title=f'./{folder}/learning_curve.png')
#
#         counter.set_postfix_str(f'D = {avg_density:.3f}, Best = {best_density:.3f}')
#     return env, best_mnznet, learning_curve


# Standard config
def make_config(w, k, l=100000, n=1, device=0, seq_path=None, save_path=''):
    vocab = list(chmap.keys())
    vocab_prob = 0.25 * torch.ones(4)
    env_arg = {
        'window_size': w,
        'kmer_size': k,
        'sequence_length': l,
        'num_sequence': n,
        'sequence_vocab': vocab,
        'sequence_vocab_probability': vocab_prob
    }
    config = {
        'env': env_arg,
        'seq_path': seq_path,
        'save_folder': save_path,
        'max_iter': 50,
        'epoch': 2000,
        'mnznet_lr': 5e-3,
        'sinenet_lr': 5e-3,
        'device': device,
        'seed': 1234,
        'eval_interval': 20,
        'batch_size': 10,
        'hidden_dim': 64,
        'sample_size': 1
    }
    return config


###############################################################
CONTROL = {
    'miniception': lambda w, k: Miniception(w, k, k0=5),
    'random': lambda w, k: MinimizersImpl(w, k),
    'pasha': lambda w, k: PASHA(w, k)
}

DIV_FUNC = {
    'l2': lambda w: lambda score, target: l2_div(score, target),
    'smooth_delta': lambda w: lambda score, target: smooth_delta_div(score, target),
    'maxpool_delta': lambda w: lambda score, target: maxpool_delta_div(score, target, w)
}


def vary_k_pasha_fix(w=14, dev=0, exp_name='chrx'):
    k_values = [6, 8, 10, 12, 14]
    for k in k_values:
        save_folder = f'results/{exp_name}_k{k}_w{w}'
        comparison = torch.load(f'{save_folder}/comparison.pth')
        comparison['pasha'] = control(w, k, dev, seq=exp_name, method='pasha')
        print(w, k, comparison['pasha'])
        torch.save(comparison, f'{save_folder}/comparison.pth')


def vary_k(w=14, dev=0, exp_name='chrx', compare=True):
    #k_values = [6, 8, 10]
    k_values = [12, 14]
    for k in k_values:
        comparison = {}
        density, save_folder = exp(w, k, dev, seq=exp_name)
        if compare:
            comparison['mnznet'] = density
            for ctrl in CONTROL.keys():
                density = control(w, k, dev, seq=exp_name, method=ctrl)
                comparison[ctrl] = density
            torch.save(comparison, f'{save_folder}/comparison.pth')


def vary_w_pasha_fix(k=13, dev=0, exp_name='chrx'):
    w_values = [85]
    for w in w_values:
        save_folder = f'results/{exp_name}_k{k}_w{w}'
        comparison = torch.load(f'{save_folder}/comparison_w.pth')
        comparison['pasha'] = control(w, k, dev, seq=exp_name, method='pasha')
        print(w, k, comparison['pasha'])
        torch.save(comparison, f'{save_folder}/comparison_w.pth')


def vary_w(k=13, dev=0, exp_name='chrx'):
    #w_values = [10, 25, 40, 55, 70, 85]
    w_values = [70, 85]
    for w in w_values:
        comparison = {}
        density, save_folder = exp(w, k, dev, seq=exp_name)
        comparison['mnznet'] = density
        for ctrl in CONTROL.keys():
            density = control(w, k, dev, seq=exp_name, method=ctrl)
            comparison[ctrl] = density
        torch.save(comparison, f'{save_folder}/comparison_w.pth')


def compare_divergence(seq='chr1', dev=0):
    w_val = [14]
    k_val = [12, 14]
    for w in w_val:
        for k in k_val:
            comparison = {}
            for df in DIV_FUNC.keys():
                comparison[df], save_folder = exp(w, k, dev, seq, df=df, save_path='results_set2')
                torch.save(comparison, f'{save_folder}/comparison.pth')


def exp(w=14, k=25, dev=0, seq='chr1', df='maxpool_delta', save_path='results'):
    config = make_config(w, k, l=500 * (w + k), device=dev, seq_path=f'{seq}.seq', save_path=f'{save_path}/{seq}_k{k}_w{w}')
    config['div_func'] = DIV_FUNC[df](w)
    env, mnznet, learning_curve, best_density = deep_minimizer(config)
    # export_artifacts(env, mnznet, config)
    torch.save(learning_curve, f'{config["save_folder"]}/summary.pth')
    del mnznet, env, learning_curve
    torch.cuda.empty_cache()
    return best_density, config['save_folder']


def control(w=14, k=25, dev=0, seq='chr1', method='miniception'):
    config = make_config(w, k, l=500 * (w + k), device=dev, seq_path=f'{seq}.seq', save_path=f'results/{seq}_k{k}_w{w}')
    f = open(f"./{config['seq_path']}", 'r')
    lines = f.readlines()
    seq = ''.join([line.strip() for line in lines])
    mm = CONTROL[method](w, k)
    density = calc_selected_locs(seq, mm)
    del seq, mm, lines
    return density * (w + 1)


if __name__ == '__main__':
    exp(w=13, k=8, dev=0, seq='chrXC', save_path='results_set3')




