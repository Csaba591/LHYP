from os.path import join as pjoin
import os
import sys
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import random

# go up one level
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset_generator import Patient  

class SADataset(Dataset):
    def __init__(self, path_to_pickles, patient_ids):
        self.path = pjoin(path_to_pickles, 'sa')
        self.data = []
        self.num_frames_per_slice = 6
        
        self.transforms = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.25)
        ])
        
        self._load_pickles(patient_ids)
    
    def _transform(self, image_tensor, ROI, padding=2):
        img_h, img_w = image_tensor.shape[-1], image_tensor.shape[-2]
        cutout = self._calculate_roi_cutout(img_w, img_h, ROI, padding)
        
        image_tensor = TF.crop(image_tensor, *cutout)
        image_tensor = TF.resize(image_tensor, [128, 128])
        image_tensor = self.transforms(image_tensor)
        return image_tensor
    
    def _calculate_roi_cutout(self, img_h, img_w, ROI, padding):
        top = max(0, ROI['top'] - padding)
        left = max(0, ROI['left'] - padding)
        bottom = min(img_h, ROI['bottom'] + padding)
        right = min(img_w, ROI['right'] + padding)

        height = bottom - top
        width = right - left
        
        return top, left, height, width
        
    def _load_pickles(self, ids):
        means = [0]*len(ids)
        
        N = 0
        for i, ID in enumerate(ids):
            path = pjoin(self.path, ID+'.pickle')
            with open(path, 'rb') as dump:
                patient = pickle.load(dump)
                images = np.array(patient.images)
                # images /= 255
                means[i] = images.mean()
                N += len(images) * self.num_frames_per_slice
                patient.images = images
                self.data.append(patient)
              
        # calculate mean across all images
        # https://stats.stackexchange.com/a/420075
        global_mean = 0
        for m, p in zip(means, self.data):
            global_mean += (len(p.images) / N) * m
        
        # calculate std
        sum_of_deviations = 0
        K = 0
        for p in self.data:
            sum_of_deviations += np.square(p.images - global_mean).sum()
            K += p.images.size
        
        variance = sum_of_deviations / K
        std = np.sqrt(variance)
        
        # standardize
        # for p in self.data:
        #     p.images = (p.images - global_mean) / std
        #     print(p.images.mean(), p.images.std())
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        patient = self.data[index]
        
        start_indices = list(range(0, len(patient.images), self.num_frames_per_slice))
        random_slice = random.choice(start_indices)
        
        images = patient.images[random_slice:random_slice+self.num_frames_per_slice]
        
        image_tensors = [TF.to_tensor(im) for im in images]
        
        transformed_images = [
            self._transform(it, patient.ROI, padding=5) 
            for it in image_tensors]
        
        transformed_images_tensor = torch.cat(transformed_images)
        
        label = patient.label.lower()
        if 'u_' in label or 'normal' in label:
            label_tensor = torch.Tensor([0])
        else: label_tensor = torch.Tensor([1])
        
        return (transformed_images_tensor, label_tensor)
    
    @property
    def num_channels(self):
        return self.num_frames_per_slice
    
class SALEDataset(Dataset):
    def __init__(self, path_to_pickles, patient_ids):
        self.path = pjoin(path_to_pickles, 'sale')
        self.num_frames_per_sequence = None
        self.data = []
        
        self.transforms = transforms.Compose([
            transforms.RandomCrop(int(128*0.8)),
            transforms.RandomRotation(180),
            transforms.Resize([128, 128]),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.25)
        ])
        
        self._load_pickles(patient_ids)
    
    def _transform(self, image_tensor):
        image_tensor = TF.resize(image_tensor, [128, 128])
        image_tensor = self.transforms(image_tensor)
        return image_tensor
    
    def _load_pickles(self, ids):
        for ID in ids:
            path = pjoin(self.path, ID+'.pickle')
            with open(path, 'rb') as dump:
                patient = pickle.load(dump)
                self.data.append(patient)
                
                if self.num_frames_per_sequence is None or len(patient.images) < self.num_frames_per_sequence:
                    self.num_frames_per_sequence = len(patient.images)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        patient = self.data[index]
        
        if len(patient.images) == self.num_frames_per_sequence:
            images = patient.images
        else:
            # select evenly spaced elements to exclude
            indices = list(range(len(patient.images)))
            surplus = len(patient.images)-self.num_frames_per_sequence
            exclude = np.round(np.linspace(0, len(indices)-1, surplus)).astype(np.uint8)
            
            images = [patient.images[i] for i in indices if i not in exclude]
        
        image_tensors = [TF.to_tensor(im) for im in images]
        transformed_images = [
            self._transform(it) 
            for it in image_tensors]
        
        transformed_images_tensor = torch.cat(transformed_images)
        
        label = patient.label.lower()
        if 'u_' in label or 'normal' in label:
            label_tensor = torch.Tensor([0])
        else: label_tensor = torch.Tensor([1])
        
        return (transformed_images_tensor, label_tensor)
    
    @property
    def num_channels(self):
        return self.num_frames_per_sequence

    @num_channels.setter
    def num_channels(self, x):
        self.num_frames_per_sequence = x
