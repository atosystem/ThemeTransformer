"""Theme Transformer Arguments

    Author: Ian Shih
    Email: yjshih23@gmail.com
    Date: 2021/11/03
    
"""
import argparse
parser = argparse.ArgumentParser(description='Theme Transformer Args')

parser.add_argument('--lr', type=float, default=0.0002,
                    help='initial learning rate')
parser.add_argument('--warmup_step', type=int, default=400,
                    help='upper epoch limit')
parser.add_argument('--lr_min', type=float, default=1e-5,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=3,
                    help='gradient clipping')

parser.add_argument('--max_step', type=int, default=3200000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--max_len', type=int, default=512,
                    help='number of tokens to predict')
parser.add_argument('--seed', type=int, default=1000,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--restart_point', type=str, default='',
                    help='restart_point')

args = parser.parse_args()


