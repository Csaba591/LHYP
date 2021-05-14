import os
import sys
import random
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
import torch.nn.functional as F
from dataset import *
from ml_utils import *
import nets
import json

# go up one level
sys.path.append('..')

from dataset_generator import Patient

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

assert torch.cuda.is_available(), 'GPU unavailable'

device = torch.device('cuda')

# hyperparams
input_shape = (224, 224)
validate_every = 4


class EarlyStopping:
    def __init__(self, patience, delta, model_save_path, verbose=True):
        self.patience = patience
        self.delta = delta
        self.patience_counter = 0
        self.best_loss = None
        self.path = model_save_path
        self.verbose = verbose

    def __call__(self, loss, model, optimizer, lr_scheduler, epoch):
        loss = abs(loss)
        if self.best_loss is None or loss < self.best_loss - self.delta:
            self.best_loss = loss
            self._save_checkpoint(model, optimizer, lr_scheduler, epoch)
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            return True
        return False

    def _save_checkpoint(self, model, optimizer, lr_scheduler, epoch):
        os.makedirs(self.path, exist_ok=True)
        path = os.path.join(self.path, model.save_path)
        if self.verbose:
            eprint(
                f'Saving model checkpoint with val loss: {self.best_loss} to {path}')

        torch.save({
            'model_name': model.name,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'val_loss': self.best_loss
        }, path)


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            lr_scheduler,
            train_ds,
            val_ds,
            num_epochs,
            batch_size,
            validate_every,
            checkpoint_saver=None,
            model_load_path=None):

        self.model = model
        self.optim = optimizer
        self.batch_size = batch_size
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.criterion = torch.nn.BCELoss()
        # self.criterion = BinaryFocalLossWithLogits(alpha=0.25, reduction='mean')
        self.validate_every = validate_every
        self.checkpoint_saver = checkpoint_saver
        self.epoch = 1
        self.logger = Logger(self.model.name)
        self.model_load_path = model_load_path

        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs

        if model_load_path is not None:
            self._load_model(model_load_path)

        self._init_dataloaders(train_ds, val_ds, batch_size)

    def _print_progress(self, epoch, total_epochs, learn_loss, val_loss):
        eprint(
            f'Epoch [{epoch}/{total_epochs}] - train loss: {learn_loss} - val loss: {val_loss}')

    def single_batch_test(self):
        X, y = next(iter(self.train_loader))

        # move to GPU (if available)
        X = X.to(device)
        y = y.to(device)

        for epoch in range(self.num_epochs):
            # forward
            output = self.model(X)
            loss = self.criterion(output, y)

            self._print_progress(epoch, self.num_epochs, loss, None)

            # backward
            self.optim.zero_grad()
            loss.backward()

            self.optim.step()

    def train(self):
        eprint('Starting training:')
        eprint('- Epochs:', self.num_epochs)
        eprint('- Training samples (w/o aug):', len(self.train_loader)
               * self.batch_size, '=', len(self.train_loader), 'batches')
        eprint('- Validation samples (w/o aug):', len(self.val_loader)
               * self.batch_size, '=', len(self.val_loader), 'batches')
        eprint('- Batch size:', self.batch_size)

        moving_avg_loss = 0
        for epoch in range(self.epoch, self.epoch+self.num_epochs):
            moving_avg_loss += self.learn()
            if epoch % validate_every == 0:
                moving_avg_loss /= validate_every
                validation_loss = self.validate()

                self._print_progress(
                    epoch, self.epoch+self.num_epochs-1, moving_avg_loss, validation_loss)
                self.logger.log(epoch, moving_avg_loss, validation_loss)
                if self.checkpoint_saver is not None and \
                    self.checkpoint_saver(
                        validation_loss, self.model, self.optim, self.lr_scheduler, epoch):

                    eprint('Stopping early at val loss:', self.checkpoint_saver.best_loss)
                    return
                self.lr_scheduler.step(validation_loss)

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
            total_loss += loss.mean().item()

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
                total_loss += self.criterion(output, y).mean().item()

        self.model.train()
        return total_loss / count

    def eval(self, data_loader=None):
        if data_loader is None:
            data_loader = self.val_loader

        eprint('[Evaluation]')
        if self.model_load_path is not None:
            self._load_model(self.model_load_path)
        else: 
            eprint('> Couldn\'t load model as there was no load path given!')
            eprint('> Running with latest in-memory weights.')
        
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
                    truth = real_y.item() >= 0.5
                    prediction = out_tensor.item() >= 0.5
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

        if TP > 0:
            avgs['TP'] /= TP
        if FP > 0:
            avgs['FP'] /= FP
        if TN > 0:
            avgs['TN'] /= TN
        if FN > 0:
            avgs['FN'] /= FN
        eprint('Average outputs for validation (0.0 = no such case):')
        eprint(avgs)

        eprint(TP, FP, TN, FN)
        acc = (TP + TN) / (TP+FP+TN+FN)
        eprint(f'Accuracy: {round(acc, 5)}')

        self.logger.log_stats(TP, FP, TN, FN)
        self.logger.close()

        self.model.train()

    def _load_model(self, model_path):
        path = os.path.join(model_path, self.model.save_path)
        if not os.path.exists(path):
            return

        data = torch.load(path)

        self.model.load_state_dict(data['model_state_dict'])
        self.optim.load_state_dict(data['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(data['lr_scheduler_state_dict'])

        model_name = data['model_name']
        eprint(f'Loaded model \"{model_name}\"!')

        if 'epoch' in data and 'val_loss' in data:
            epoch, loss = data['epoch'], data['val_loss']
            self.epoch = epoch + 1
            if self.checkpoint_saver is not None:
                self.checkpoint_saver.best_loss = loss
            eprint(f'Model was saved at epoch {epoch} with val loss: {loss}')

    def _init_dataloaders(self, train_ds, val_ds, batch_size):
        self.train_loader = DataLoader(
            train_ds, batch_size, sampler=train_ds.sampler)
        self.val_loader = DataLoader(
            val_ds, batch_size, shuffle=False)

def get_generator(cross_validating, axis, path, train_ids, num_test_channels, k):
    if cross_validating:
        dataset_generator = k_fold_train_val_sets(
            axis, k, path, train_ids, num_test_channels)
    else:
        dataset_generator = train_val_sets(
            axis, 0.15, path, train_ids, num_test_channels)
    
    return dataset_generator

if __name__ == '__main__':
    seed_everything()

    with open('param_config.json', 'r') as config:
        params = json.load(config)
        num_epochs = params['epochs']
        axis = params['axis']
        cross_validating = params['CV']
        variant = params['variant']

    path = os.path.join('..', '..', 'pickled_samples')

    train_ids, test_ids = train_test_split_ids(
        axis, path, f'test_{axis}.split', test_percent=0.13)

    num_test_channels = None if axis == 'sa' else SALEDataset(
        path, test_ids).num_channels

    model_save_path = 'saved_models'
    if cross_validating: model_save_path = os.path.join(model_save_path, 'CV')

    k = 8 if cross_validating else None

    model_variant = '' if len(variant) == 0 else '_' + variant

    for batch_size in params["batch_size"]:
        for learning_rate in params["learning_rate"]:
            for model_type in params["model"]:
                eprint(f'Running {model_type}{model_variant} with: batch_size: {batch_size}, lr: {learning_rate}')

                dataset_generator = get_generator(
                    cross_validating, axis, path, train_ids, num_test_channels, k)
                
                for block_index, (train_ds, val_ds) in enumerate(dataset_generator):
                    if cross_validating:
                        eprint(f'[Cross validation] current val block: {block_index+1}/{k}')
                    
                    model_name = f'{model_type}{model_variant}_bs{batch_size}_lr{learning_rate}_{axis}'
                    if cross_validating:
                        model_name = f'CV{block_index+1}-{k}_' + model_name

                    net = nets.create_model(
                        model_type, train_ds.num_channels, input_shape, model_name, variant)
                    net.to(device)

                    # optim = torch.optim.Adam(
                    #     params=net.parameters(), lr=learning_rate)
                    optim = torch.optim.SGD(
                        net.parameters(), lr=learning_rate, momentum=0.9)

                    es_patience = 40 if cross_validating else 80
                    es = EarlyStopping(patience=es_patience, delta=0.0, model_save_path=model_save_path)
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optim, patience=es_patience//4, verbose=True)

                    trainer = Trainer(
                        model=net,
                        optimizer=optim,
                        lr_scheduler=lr_scheduler,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        validate_every=validate_every,
                        checkpoint_saver=es,
                        model_load_path=model_save_path
                    )

                    trainer.train()
                    trainer.eval()

    create_training_stats_summary()
