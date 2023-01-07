import argparse
import yaml

def create_yaml_parser():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float,
                        default=2e-4)
    parser.add_argument('--batch_size', type=int,
                        default=24)
    parser.add_argument('--num_epoch', type=int,
                        default=30)
    parser.add_argument('--num_workers', type=int,
                        default=8)
    parser.add_argument('--warmup_steps', type=int,
                        default=3)
    # model param
    parser.add_argument('--num_class', type=int,
                        default=128)
    parser.add_argument('--hidden_size', type=int,
                        default=128)
    parser.add_argument('--weight_decay', type=float,
                        default=0)
    parser.add_argument('--accumulate', type=int,
                        default=1)
    parser.add_argument('--switch', action='store_true', default=True)

    parser.add_argument('--weight', type=str,
                        default=None)
    parser.add_argument('--vae_weight', type=str,
                        default=None, requred=True)
    parser.add_argument('--save_top_k', type=int,
                        default=3)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exp_name', type=str,
                        default='exp')
    parser.add_argument('--fast_dev', action='store_true')

    parser.add_argument('--t_range', type=int,
                        default=1000)
    parser.add_argument('--beta_small', type=float,
                        default=1e-4)
    parser.add_argument('--beta_large', type=float,
                        default=0.02)
    parser.add_argument('--channel', type=int,
                        default=1)
    parser.add_argument('--in_dim', type=int,
                        default=28**2)
    parser.add_argument('--n_gen', type=int,
                        default=1)

    return parser.parse_args()
