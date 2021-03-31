import sys
import os
import random
import numpy as np
import torch
from dataset import SADataset, SALEDataset

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def conv_output_shape(h, w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h_ = floor( ((h + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w_ = floor( ((w + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h_, w_

def accuracy(TP, FP, TN, FN):
    ''' Right answers / wrong answers '''
    return (TP+TN) / (TP+FP+TN+FN)

def precision(TP, FP):
    return TP / (TP + FP)

def recall(TP, FN):
    ''' True positives / all positives '''
    return TP / (TP + FN + 1e-5)

def specificity(TN, FP):
    ''' True negatives / all negatives '''
    return TN / (TN + FP + 1e-5)

def F1(recall, precision):
    return 2*(recall * precision) / (recall + precision)

def train_test_split(axis, pickles_path, test_ids_path, test_percent=0.15):
    axis_path = os.path.join(pickles_path, axis)
    all_ids = [pkl.split('.')[-2] for pkl in os.listdir(axis_path)]
    
    # get test ids
    # if the test id file exists: read them in
    # else: generate new test ids randomly according to test_percent
    if os.path.exists(test_ids_path):
        with open(test_ids_path, 'rt') as ids:
            test_ids = ids.read().splitlines()
    else:
        if test_percent > 1: test_percent /= 100
        
        # calculate split
        test_len = int(len(all_ids) * test_percent)
        
        if test_len > 0:
            # sample test ids from all ids randomly
            test_ids = random.sample(all_ids, test_len)
            
            # save test ids to file
            with open(test_ids_path, 'wt') as test_ids_file:
                test_ids_out = '\n'.join(test_ids) + '\n'
                test_ids_file.write(test_ids_out)
        else: test_ids = []

    train_ids = [ID for ID in all_ids if ID not in test_ids]

    if len(test_ids) > 0:
        test_ds = SADataset(pickles_path, test_ids) if axis.lower() == 'sa' else SALEDataset(pickles_path, test_ids)
    else: test_ds = None
    
    train_ds = SADataset(pickles_path, train_ids) if axis.lower() == 'sa' else SALEDataset(pickles_path, train_ids)
    
    if test_ds is not None and axis.lower() == 'sale':
        num_channels = min(test_ds.num_channels, train_ds.num_channels)
        test_ds.num_channels = num_channels
        train_ds.num_channels = num_channels
    return train_ds, test_ds
