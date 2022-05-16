import torch
import math
import random
import numpy as np

# Prune
def magnitude_prune(masking, name, mask, num_prune, weight):
    """Prunes the weights with smallest magnitude.
    """
    
    num_prune = num_prune
    num_zeros = masking.name2zeros[name]
    
    # new number of zero parameters after pruning
    new_num_zeros = math.ceil(num_zeros + num_prune)
    if num_prune == 0.0: return weight.data != 0.0

      
    omega = weight
    x, idx = torch.sort(torch.abs(omega.data.view(-1)))
    
    # adjust the mask
    mask.data.view(-1)[idx[:new_num_zeros]] = 0.0
    return mask

# Growth
def momentum_growth(masking, name, new_mask, num_growth, weight):
    """Grows weights in places where the momentum is largest.
    """
    miu = masking.get_momentum_for_weight(weight)
    if miu.dtype == torch.float16:
        miu = miu*(new_mask==0).half()
    else:
        miu = miu*(new_mask==0).float()
    
    # determine the indice of parameters to be pruned: idx[:num_growth]
    y, idx = torch.sort(torch.abs(miu.data.view(-1)), descending=True)
    new_mask.data.view(-1)[idx[:num_growth]] = 1.0

    return new_mask

# Fix Random Seed
def init_seed(seed=None):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

prune_funcs = {}
prune_funcs['magnitude'] = magnitude_prune

growth_funcs = {}
growth_funcs['momentum'] = momentum_growth
