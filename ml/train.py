import os
import sys
import random
from dataset import SADataset, SALEDataset
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ml_utils import *
import nets

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
    'train': 0.5, 
    'val': 0.5, 
    'test': 0.0
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
        dataset, 
        early_stopping=None,
        model_load_path=None):
        
        self.ds = dataset
        self.model = model
        self.optim = optimizer
        self.criterion = F.binary_cross_entropy
        self.early_stopping = early_stopping
        self.epoch = 0
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0
        self.logger = Logger(self.model.name)
        
        if model_load_path is not None:
            self._load_model(model_load_path)
        
        self._init_dataset()
    
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
        eprint('- Training samples (w/o aug):', len(self.train_set))
        eprint('- Validation samples (w/o aug):', len(self.val_set))
        eprint('- Batch size:', batch_size)
        
        for epoch in range(self.epoch, self.epoch+num_epochs):
            learning_loss = self.learn()
            if epoch % validate_every == 0:
                validation_loss = self.validate()
                # self._print_progress(epoch, self.epoch+num_epochs, learning_loss, validation_loss)
                self.logger.log(epoch, learning_loss, validation_loss)
                if self.early_stopping is not None and \
                    self.early_stopping(validation_loss, self.model, self.optim, epoch):
                    
                    eprint('Stopping early at val loss:', validation_loss.item())
                    return
        
        self.logger.close()
            
    def learn(self):
        self.model.train()
        
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
                output = net(X)
                total_loss += self.criterion(output, y)
        
        return total_loss / count
    
    def _print_eval(self):
        eprint(f'True pos:\t{self.TP}')
        eprint(f'True neg:\t{self.TN}')
        eprint(f'False pos:\t{self.FP}')
        eprint(f'False neg:\t{self.FN}')
        eprint('Accuracy:', accuracy(self.TP, self.FP, self.TN, self.FN), sep='\t')
        rec = recall(self.TP, self.FN)
        eprint('Recall:', round(rec, 4), sep='\t\t')
        spec = specificity(self.TN, self.FP)
        eprint('Specificity:', round(spec, 4), sep='\t')
        prec = precision(self.TP, self.FP)
        eprint('Precision:', round(prec, 4), sep='\t')
        F1_score = F1(rec, prec)
        eprint('F1:', round(F1_score, 4), sep='\t\t')
         
    def eval(self):
        self.model.eval()
        self.TP, self.FP, self.TN, self.FN = 0, 0, 0, 0
        
        with torch.no_grad():
            for X, y in self.val_loader:
                # move to GPU (if available)
                X = X.to(device)
                y = y.to(device)
                
                output = net(X)
                
                for real_y, out_tensor in zip(y, output):
                    truth = real_y.item() > 0.5
                    prediction = out_tensor.item() > 0.5
                    if prediction == truth:
                        if truth is True:
                            self.TP += 1
                        else: self.TN += 1
                    else:
                        if truth is False:
                            self.FP += 1
                        else: self.FN += 1
        
        self.logger.log_stats(self.TP, self.FP, self.TN, self.FN)
        # self._print_eval()
      
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
            if self.early_stopping is not None:
                self.early_stopping.best_loss = loss
            eprint(f'Model was saved at epoch {epoch} with val loss: {loss}')
        
    def _init_dataset(self):
        val_len = int(len(self.ds) * splits['val'])
        train_len = len(self.ds) - val_len
        assert val_len > 0, '_init_dataset(): val_len can\'t be 0!'
        assert train_len > 0, '_init_dataset(): train_len can\'t be 0!'
        
        lengths = [train_len, val_len]
        
        self.train_set, self.val_set = random_split(self.ds, lengths)
        
        self.train_loader = DataLoader(self.train_set, batch_size, True)
        self.val_loader = DataLoader(self.val_set, batch_size, True)

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
    
    path = os.path.join('..', '..', 'pickled_samples')
    train_ds, test_ds = train_test_split('sa', path, 'test2.split', splits['test'])
    
    # for images, label in train_ds:
    #     eprint(images.shape, images.mean(), images.std())
    
    net = nets.SimpleCNN(train_ds.num_channels, input_shape, 'SimpleCNN')
    # net = nets.Linear(train_ds.num_channels, input_shape, 'SimpleLinear')
    net.to(device)
    
    optim = torch.optim.Adam(params=net.parameters(), lr=lr)
    
    model_save_path = 'saved_models'
    es = EarlyStopping(patience=num_epochs, delta=0.0, model_save_path=model_save_path)

    trainer = Trainer(net, optim, train_ds, es, model_save_path)
    
    trainer.train()
