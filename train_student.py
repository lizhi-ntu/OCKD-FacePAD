import argparse
from src.algo import train_student
from src.funcs import init_seed

# Please customize the script below >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
source_set = 'casia'
target_set = 'client'
density = 0.1
mask = 1
setting = 'ideal'
tail = 'train_{}_student_{}'.format(setting, int(density * 100))
# Please customize the script above <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

parser = argparse.ArgumentParser()
parser.add_argument('--tail', type=str, default=tail, help='name of experiment')
parser.add_argument('--source_set', type=str, default=source_set, help='name of dataset')
parser.add_argument('--target_set', type=str, default=target_set, help='name of dataset')
parser.add_argument('--ptc', type=str, default='adaptation', help='')
parser.add_argument('--mask', type=int, default=mask, help='client ID')
parser.add_argument('--setting', type=str, default=setting, help='mode of evaluation settings: ideal or challenging')
parser.add_argument('--density', type=float, default=density, help='density of the student network')
parser.add_argument('--teacher_path', type=str, default='checkpoint/teacher/{}/teacher.pt'.format(source_set), help='checkpoint of the teacher network')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--bs', type=int, default=25, help='batch size')
parser.add_argument('--nw', type=int, default=16, help='number of workers')
parser.add_argument('--period', type=int, default=60, help='period (in terms of iterations) for parameter regrowth and in-training model test')
parser.add_argument('--epoch', type=int, default=3000, help='maximum of training epoch')
parser.add_argument('--sps', type=int, default=128, help='size of input image')
parser.add_argument('--seed', type=int, default=1, help='random seed')

opt = parser.parse_args()
print(opt)

init_seed(seed=opt.seed)

train_student(opt)
