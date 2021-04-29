import os
import sys
import random
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import torch.nn.functional as F
from dataset import *
from ml_utils import *
import nets
import json

# go up one level
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('..')

from dataset_generator import Patient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
input_shape = (128, 128)
num_epochs = 50
validate_every = 4
lr = 0.001
batch_size = 4
splits = {
    'train': 0.7, 
    'val': 0.15, 
    'test': 0.15
}

class EarlyStopping:
    def __init__(self, patience, delta, model_save_path, verbose=True):
        self.patience = patience
        self.delta = delta
        self.patience_counter = 0
        self.best_loss = None
        self.path = model_save_path
        self.verbose = verbose
        
    def __call__(self, val_loss, model, optimizer, epoch):
        loss = val_loss.item()
        if self.best_loss is None or loss <= self.best_loss - self.delta:
            self.best_loss = loss
            self._save_checkpoint(model, optimizer, epoch)
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            return True
        return False
    
    def _save_checkpoint(self, model, optimizer, epoch):
        os.makedirs(self.path, exist_ok=True)
        path = os.path.join(self.path, model.save_path)
        if self.verbose:
            eprint(f'Saving model checkpoint with val loss: {self.best_loss} to {path}')
        
        torch.save({
            'model_name': model.name,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': self.best_loss
        }, path)

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        train_ds,
        val_ds, 
        batch_size, 
        validate_every, 
        checkpoint_saver=None,
        model_load_path=None):
        
        self.model = model
        self.optim = optimizer
        self.batch_size = batch_size
        self.criterion = F.binary_cross_entropy
        self.validate_every = validate_every
        self.checkpoint_saver = checkpoint_saver
        self.epoch = 1
        self.logger = Logger(self.model.name)
        
        if model_load_path is not None:
            self._load_model(model_load_path)
        
        self._init_dataloaders(train_ds, val_ds, batch_size)
    
    def _print_progress(self, epoch, total_epochs, learn_loss, val_loss):
        eprint(f'Epoch [{epoch}/{total_epochs}] - train loss: {learn_loss.item()}', end='')
        if val_loss is not None:
            eprint(f' - val loss: {val_loss.item()}')
        else: eprint()
     
    def single_batch_test(self):
        X, y = next(iter(self.train_loader))
        
        # move to GPU (if available)
        X = X.to(device)
        y = y.to(device)
        
        for epoch in range(num_epochs):
            # forward
            output = self.model(X)
            loss = self.criterion(output, y)
            
            self._print_progress(epoch, num_epochs, loss, None)
            
            # backward
            self.optim.zero_grad()
            loss.backward()
            
            self.optim.step()
    
    def train(self):
        eprint('Starting training:')
        eprint('- Epochs:', num_epochs)
        eprint('- Training samples (w/o aug):', len(self.train_loader)*self.batch_size, '=', len(self.train_loader), 'batches')
        eprint('- Validation samples (w/o aug):', len(self.val_loader)*self.batch_size, '=', len(self.val_loader), 'batches')
        eprint('- Batch size:', self.batch_size)
        
        moving_avg_loss = 0
        for epoch in range(self.epoch, self.epoch+num_epochs):
            learning_loss = self.learn()
            moving_avg_loss += learning_loss.item()
            if epoch % validate_every == 0:
                moving_avg_loss /= validate_every
                validation_loss = self.validate()
                
                self._print_progress(epoch, self.epoch+num_epochs, learning_loss, validation_loss)
                self.logger.log(epoch, moving_avg_loss, validation_loss.item())
                if self.checkpoint_saver is not None and \
                    self.checkpoint_saver(validation_loss, self.model, self.optim, epoch):
                    
                    eprint('Stopping early at val loss:', validation_loss.item())
                    return
        
        self.logger.close()
            
    def learn(self):
        count = 0
        total_loss = 0.0
        for X, y in self.train_loader:
            # move to GPU (if available)
            X = X.to(device)
            y = y.to(device)
            
            count += 1
            
            # forward
            output = self.model(X)
            loss = self.criterion(output, y)
            total_loss += loss
            
            # backward
            self.optim.zero_grad()
            loss.backward()
            
            self.optim.step()
            
        return total_loss / count
    
    def validate(self):
        self.model.eval()
        
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                # move to GPU (if available)
                X = X.to(device)
                y = y.to(device)
            
                count += 1
                
                # forward
                output = self.model(X)
                total_loss += self.criterion(output, y)
        
        self.model.train()
        return total_loss / count
         
    def eval(self, data_loader=None):
        if data_loader is None:
            data_loader = self.val_loader
        
        self.model.eval()
        TP, FP, TN, FN = 0, 0, 0, 0
        
        avgs = {'TP': 0.0, 'FP': 0.0, 'TN': 0.0, 'FN': 0.0}
        
        with torch.no_grad():
            for X, y in data_loader:
                # move to GPU (if available)
                X = X.to(device)
                y = y.to(device)
                
                output = self.model(X)
                
                for real_y, out_tensor in zip(y, output):
                    truth = real_y.item() > 0.5
                    prediction = out_tensor.item() > 0.5
                    if prediction == truth:
                        if truth is True:
                            TP += 1
                            avgs['TP'] += out_tensor.item()
                        else: 
                            TN += 1
                            avgs['TN'] += out_tensor.item()
                    else:
                        if truth is False:
                            FP += 1
                            avgs['FP'] += out_tensor.item()
                        else: 
                            FN += 1
                            avgs['FN'] += out_tensor.item()
        
        if TP > 0: avgs['TP'] /= TP
        if FP > 0: avgs['FP'] /= FP
        if TN > 0: avgs['TN'] /= TN
        if FN > 0: avgs['FN'] /= FN
        eprint('Average outputs for validation (0.0 = no such case):')
        eprint(avgs)
        
        self.logger.log_stats(TP, FP, TN, FN)
        self.model.train()
      
    def _load_model(self, model_path):
        path = os.path.join(model_path, self.model.save_path)
        if not os.path.exists(path):
            return
        
        data = torch.load(path)
        
        self.model.load_state_dict(data['model_state_dict'])
        self.optim.load_state_dict(data['optimizer_state_dict'])
        
        model_name = data['model_name']
        eprint(f'Loaded model \"{model_name}\"!')
        
        if 'epoch' in data and 'val_loss' in data:
            epoch, loss = data['epoch'], data['val_loss']
            self.epoch = epoch + 1
            if self.checkpoint_saver is not None:
                self.checkpoint_saver.best_loss = loss
            eprint(f'Model was saved at epoch {epoch} with val loss: {loss}')
        
    def _init_dataloaders(self, train_ds, val_ds, batch_size):
        self.train_loader = DataLoader(train_ds, batch_size, sampler=train_ds.sampler)
        self.val_loader = DataLoader(val_ds, batch_size, sampler=val_ds.sampler)

def save_model(model, optimizer, path):
    full_path = os.path.join(path, model.save_path)
    os.makedirs(path, exist_ok=True)
    
    torch.save({
        'model_name': model.name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, full_path)

if __name__ == '__main__':
    # seed_everything(2222222222)
    
    with open('param_config.json', 'r') as config:
        params = json.load(config)
        
    axis = 'sa'
    path = os.path.join('..', '..', 'pickled_samples')
    model_save_path = 'saved_models'
                
    train_ids, test_ids = train_test_split_ids(axis, path, 'test2.split', splits['test'])
    
    k = 10
    dataset_generator = k_fold_train_val_sets(axis, k, path, train_ids)
    
    for block_index, (train_ds, val_ds) in enumerate(dataset_generator):
        eprint(f'[Cross validation] current val block: {block_index+1}/{k}')
        for batch_size in params["batch_size"]:
            for learning_rate in params["learning_rate"]:
                eprint(f'Running with: batch_size: {batch_size}, lr: {learning_rate}')
                
                net = nets.BasicCNN(train_ds.num_channels, input_shape, 'BasicCNN')
                net.to(device)
                
                optim = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
                
                es = EarlyStopping(patience=num_epochs, delta=0.0, model_save_path=model_save_path)

                trainer = Trainer(
                    model=net,
                    optimizer=optim,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    batch_size=batch_size,
                    validate_every=validate_every,
                    checkpoint_saver=es,
                    model_load_path=model_save_path
                )
                
                trainer.train()
                trainer.eval()
