import argparse
from src.algo import evaluation_student


# Please customize the script below >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
source_set = 'casia'
target_set = 'client'
mask = 1
density = 0.1
setting = 'ideal'
# Please customize the script above <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


parser = argparse.ArgumentParser()
parser.add_argument('--source_set', type=str, default=source_set, help='name of dataset')
parser.add_argument('--target_set', type=str, default=target_set, help='name of dataset')
parser.add_argument('--mask', type=int, default=mask, help='')
parser.add_argument('--tail', type=str, default='{}_{}_{}_{}_{}'.format(setting, source_set, target_set, mask, int(density*100)))
parser.add_argument('--teacher_path', type=str, default='checkpoint/teacher/{}/teacher.pt'.format(source_set))
parser.add_argument('--student_path', type=str, default='checkpoint/student_{}/{}/{}/student_{}.pt'.format(int(density*100), source_set, target_set, mask))
parser.add_argument('--setting', type=str, default=setting, help='')
parser.add_argument('--ptc', type=str, default='grand', help='')
parser.add_argument('--seed', type=int, default=1, help='')
parser.add_argument('--bs', type=int, default=30, help='')
parser.add_argument('--nw', type=int, default=16, help='')
parser.add_argument('--sps', type=int, default=128, help='')
parser.add_argument('--epoch', type=int, default=0, help='')
opt = parser.parse_args()
print(opt)

evaluation_student(opt)
