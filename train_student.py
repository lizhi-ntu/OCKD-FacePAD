import argparse
from src.algo import train_student
from src.funcs import init_seed

#==================================================================================
source_set = 'casia'
target_set = 'client'
density = 0.1
mask = 10
setting = 'ideal'
tail = '{}_student_{}'.format(setting, int(density * 100))
#==================================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--tail', type=str, default=tail, help='')
parser.add_argument('--source_set', type=str, default=source_set, help='name of dataset')
parser.add_argument('--target_set', type=str, default=target_set, help='name of dataset')
parser.add_argument('--ptc', type=str, default='adaptation', help='')
parser.add_argument('--mask', type=int, default=mask, help='')
parser.add_argument('--setting', type=str, default=setting, help='')
parser.add_argument('--density', type=str, default=density, help='')
parser.add_argument('--teacher_path', type=str, default='checkpoint/teacher/{}/teacher.pt'.format(source_set), help='')
parser.add_argument('--lr', type=float, default=0.0001, help='')
parser.add_argument('--bs', type=int, default=25, help='')
parser.add_argument('--nw', type=int, default=16, help='')
parser.add_argument('--period', type=int, default=60, help='')
parser.add_argument('--epoch', type=int, default=3000, help='')
parser.add_argument('--sps', type=int, default=128, help='')
parser.add_argument('--seed', type=int, default=1, help='')

opt = parser.parse_args()
print(opt)

init_seed(seed=opt.seed)

train_student(opt)
