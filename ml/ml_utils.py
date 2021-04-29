import sys
import os
from math import nan, isnan
import random
import numpy as np
import torch
from dataset import SADataset, SALEDataset
import csv

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
    if TP + FP == 0: return nan
    return TP / (TP + FP)

def recall(TP, FN):
    ''' True positives / all positives '''
    if TP + FN == 0: return nan
    return TP / (TP + FN)

def specificity(TN, FP):
    ''' True negatives / all negatives '''
    if TN + FP == 0: return nan
    return TN / (TN + FP)

def balanced_accuracy(recall, specificity):
    TPR = recall
    TNR = specificity
    if isnan(TPR) or isnan(TNR): return nan
    return (TPR + TNR) / 2

def F1(recall, precision):
    if isnan(recall) or isnan(precision): return nan
    if recall + precision == 0: return nan
    return 2*(recall * precision) / (recall + precision)

def train_test_split_ids(axis, pickles_path, test_ids_path, test_percent=0.15):
    '''
    Split all patient ids into train and test ids.
    For SALEDataset: make sure to set 
    num_channels to min(test_ds.num_channels, train_ds.num_channels)
    '''
    axis_path = os.path.join(pickles_path, axis)
    all_ids = set(pkl.split('.')[-2] for pkl in os.listdir(axis_path))
    
    # get test ids
    # if the test id file exists: read them in
    # else: generate new test ids randomly according to test_percent
    if os.path.exists(test_ids_path):
        with open(test_ids_path, 'rt') as ids:
            test_ids = set(ids.read().splitlines())
    else:
        if test_percent > 1: test_percent /= 100
        
        # calculate split
        test_len = int(len(all_ids) * test_percent)
        assert test_len > 0, '0 patients selected for testing! test_percent too small'
        
        # sample test ids from all ids randomly
        test_ids = set(random.sample(all_ids, test_len))
        
        # save test ids to file
        with open(test_ids_path, 'wt') as test_ids_file:
            test_ids_out = '\n'.join(test_ids)
            test_ids_file.write(test_ids_out)
        
    train_ids = all_ids - test_ids
    
    return train_ids, test_ids

class Logger:
    def __init__(self, model_name):
        self.model_name = model_name
        self.file = None
        self.first_write = True
        self._open()

    def log(self, epoch, train_loss, val_loss):
        self.writer.writerow([epoch, train_loss, val_loss])
        
    def log_stats(self, TP, FP, TN, FN):
        path = os.path.join('training_logs', f'{self.model_name}_stats.csv')
        first_write = not os.path.isfile(path)
        
        stats_file = open(path, 'a', newline='')
        writer = csv.writer(stats_file, delimiter=',')
        if first_write:
            # write header
            writer.writerow(['TP', 'FP', 'TN', 'FN', 'accuracy', 'balanced_accuracy', 'recall', 'specificity', 'precision', 'F1'])
        
        acc = accuracy(TP, FP, TN, FN)
        rec = recall(TP, FN)
        spec = specificity(TN, FP)
        prec = precision(TP, FP)
        b_acc = balanced_accuracy(rec, spec)
        F1_score = F1(rec, prec)
        writer.writerow([TP, FP, TN, FN, acc, b_acc, rec, spec, prec, F1_score])

        stats_file.close()
    
    def _open(self):
        if self.file is not None and not self.file.closed:
            return
        
        os.makedirs('training_logs', exist_ok=True)
        
        path = os.path.join('training_logs', f'{self.model_name}_loss.csv')
        first_write = not os.path.isfile(path)
        self.file = open(path, 'a', newline='')
        self.writer = csv.writer(self.file, delimiter=',')
        if first_write:
            # write header
            self.writer.writerow(['epoch', 'train loss', 'val loss'])
        
       
    def close(self):
        self.file.close()

def k_fold_train_val_sets(axis, k, path_to_pickles, patient_ids, sale_test_num_channels=None):
    assert axis.lower() in ['sa', 'sale'], f'Wrong axis: {axis}'

    axis = axis.lower()
    if axis == 'sa': assert sale_test_num_channels is None, 'sale_test_num_channels should not be specified for SA'
    else: assert type(sale_test_num_channels) is int, 'sale_test_num_channels should be specified (as int) for SALE'
    
    block_size = len(patient_ids) // k
    current_block = 1
    for val_block_start in range(0, k*block_size, block_size):
        if current_block < k:
            val_block_end = val_block_start + block_size - 1
        else: val_block_end = len(patient_ids) - 1
    
        train_ids, val_ids = [], []
        for index, ID in enumerate(patient_ids):
            if index < val_block_start or index > val_block_end:
                train_ids.append(ID)
            else: val_ids.append(ID)
    
        current_block += 1
    
        if axis == 'sa':
            train_ds = SADataset(path_to_pickles, train_ids)
            val_ds = SADataset(path_to_pickles, val_ids)
        else:
            train_ds = SALEDataset(path_to_pickles, train_ids)
            val_ds = SALEDataset(path_to_pickles, val_ids)
            
            min_num_channels = min(
                train_ds.num_channels, 
                val_ds.num_channels, 
                sale_test_num_channels)
            
            train_ds.num_channels = min_num_channels
            val_ds.num_channels = min_num_channels
    
        yield train_ds, val_ds
        