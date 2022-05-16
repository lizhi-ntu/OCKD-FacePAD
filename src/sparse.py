from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.optim as optim
from src.funcs import prune_funcs, growth_funcs

class CosineDecay(object):
    # Decays a pruning rate according to a cosine schedule
    
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.sgd.step()
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class Masking(object):
    # Wraps PyTorch model parameters with a sparse mask.
    
    def __init__(self, optimizer, prune_rate_decay, prune_rate, prune_mode='magnitude', growth_mode='momentum', fp16=False):    
        self.optimizer = optimizer                      # optimizer
        self.prune_rate = prune_rate                    # prune rate
        self.prune_rate_decay = prune_rate_decay        # scheduler for prune rate
        self.prune_func = prune_funcs[prune_mode]       # prune function
        self.growth_func = growth_funcs[growth_mode]    # growth function

        self.modules = []                               
        self.names = []                                 
        self.masks = {}                                 # A dictionary to store masks (indices for active and inactive parameters)
        self.name2variance = {}                         
        self.name2zeros = {}                           
        self.name2nonzeros = {}                        
        self.name2pruned = {}                           
        self.name2prune_rate = {}                       

        self.total_param = 0
        self.expect_nonzero = 0
        self.total_pruned = 0
        self.total_zero = 0
        self.total_nonzero = 0

        self.half = fp16
        self.name_to_32bit = {}

    def add_module(self, module, density):
        self.modules.append(module)                             # add on the model to be trained

        for name, tensor in module.named_parameters():
            self.names.append(name)                             
            self.masks[name] = torch.zeros_like(
                tensor, dtype=torch.float32, 
                requires_grad=False).cuda()                     
        
        print('Removing biases...')
        self.remove_weight_partial_name('bias')                 # remove bias
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)                        # remove batch norm2d
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)                        # remove batch norm1d
        
        # network and optimizer initialization
        self.init(density=density)                              

    def init(self, density):
        # optimizer initialization
        self.check_optimizer()                                                                           
        
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                
                # generate initial masks
                v = 1 - torch.rand(weight.shape)
                num_active = round(weight.numel()*density)
                x, idx = torch.sort(torch.abs(v.data.view(-1)), descending=True)
                v.data.view(-1)[idx[:num_active]] = 1.0
                v.data.view(-1)[idx[num_active:]] = 0.0
                self.masks[name][:] = v.float().data.cuda() 
                
                # count the number of expected nonzero params
                self.expect_nonzero += weight.numel()*density
                # count the number of total params
                self.total_param += weight.numel()

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.name2variance[name] = weight.numel() / float(self.total_param)   

        # set inactive parameters as 0
        self.apply_mask()                                                                              
      
        # print nonzero information
        self.print_nonzero_counts() 
        
        total_size = 0
        for name, module in self.modules[0].named_modules():
            if hasattr(module, 'weight'):
                total_size += module.weight.numel()
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    total_size += module.bias.numel()
        print('Total Model parameters:', total_size)

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total parameters after removed layers:', total_size)
        print('Total parameters under density of {0}: {1}'.format(density, density*total_size))
    
    #=========================================================================================================
    def check_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True

    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()

    def apply_mask(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]

    def param_regrowth(self):
        self.gather_statistics()
        self.adjust_prune_rate()
        total_nonzero_new = 0
        
        name2growth = self.calc_growth()
        
        # prune
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                new_mask = self.prune_func(self, name, mask, math.floor(name2growth[name]), weight)
                removed = self.name2nonzeros[name] - new_mask.sum().item()
                self.total_pruned += removed
                self.name2pruned[name] = removed
                self.masks[name][:] = new_mask

        # growth
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.bool()

                new_mask = self.growth_func(self, name, new_mask, math.floor(name2growth[name]), weight)
                new_nonzero = new_mask.sum().item()

                # update mask
                self.masks.pop(name)
                self.masks[name] = new_mask.float()
                total_nonzero_new += new_nonzero
        
        self.apply_mask()
 
    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.name2prune_rate[name] = self.prune_rate

    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2removed = {}

        self.total_pruned = 0
        self.total_nonzero = 0
        self.total_zero = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]

                self.name2nonzeros[name] = mask.sum().item()

                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]

    def calc_growth(self):
        name2growth = {}
        for name in self.name2variance:
            prune_rate = self.name2prune_rate[name]
            name2growth[name] = math.ceil(prune_rate*self.name2nonzeros[name])
    
        return name2growth

    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name+'.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name+'.weight'].shape, self.masks[name+'.weight'].numel()))
            self.masks.pop(name+'.weight')
        else:
            print('ERROR',name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                
                print(name, num_nonzeros)

        print('Prune rate: {0}\n'.format(self.prune_rate))
