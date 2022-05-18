import argparse
from src.algo import train_teacher
from src.funcs import init_seed

# Please customize the script below >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
dataset_name = 'casia'
# Please customize the script above <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

parser = argparse.ArgumentParser()
parser.add_argument('--source_set', type=str, default=dataset_name, help='name of dataset')
parser.add_argument('--target_set', type=str, default=dataset_name, help='name of dataset')
parser.add_argument('--tail', type=str, default='train_teacher', help='name of experiment')
parser.add_argument('--mask', type=int, default=0, help='client ID')
parser.add_argument('--ptc', type=str, default='grand', help='')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--bs', type=int, default=30, help='batch size')
parser.add_argument('--nw', type=int, default=16, help='number of workers')
parser.add_argument('--modality', type=str, default='rgb', help='modality of the input')
parser.add_argument('--aug', type=str, default='none', help='name of augmentation')
parser.add_argument('--period', type=int, default=400, help='period (in terms of iterations) for in-training model test')
parser.add_argument('--epoch', type=int, default=200, help='maximum of training epoch')
parser.add_argument('--sps', type=int, default=128, help='size of input image')
parser.add_argument('--seed', type=int, default=1, help='random seed')
opt = parser.parse_args()
print(opt)

init_seed(seed=opt.seed)

train_teacher(opt)
