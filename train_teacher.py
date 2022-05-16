import argparse
from src.algo import train_teacher
from src.funcs import init_seed

#====Customize Here==============
dataset_name = 'casia'
#================================

parser = argparse.ArgumentParser()
parser.add_argument('--source_set', type=str, default=dataset_name, help='name of dataset')
parser.add_argument('--target_set', type=str, default=dataset_name, help='name of dataset')
parser.add_argument('--tail', type=str, default='teacher', help='')
parser.add_argument('--mask', type=int, default=0, help='')
parser.add_argument('--ptc', type=str, default='grand', help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--bs', type=int, default=30, help='')
parser.add_argument('--nw', type=int, default=16, help='')
parser.add_argument('--modality', type=str, default='rgb', help='')
parser.add_argument('--aug', type=str, default='none', help='name of augmentation')
parser.add_argument('--period', type=int, default=400, help='')
parser.add_argument('--epoch', type=int, default=200, help='')
parser.add_argument('--sps', type=int, default=128, help='')
parser.add_argument('--seed', type=int, default=1, help='random seed')
opt = parser.parse_args()
print(opt)

init_seed(seed=opt.seed)

train_teacher(opt)
