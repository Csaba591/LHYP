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
    def __init__(self, path_to_pickles):
        self.path = pjoin(path_to_pickles, 'sa')
        self.num_frames_per_slice = 6
        self.data = []
        
        self._load_pickles()
    
    def _transform(self, image_tensor, ROI, padding=2):
        img_h, img_w = image_tensor.shape[-1], image_tensor.shape[-2]
        cutout = self._calculate_roi_cutout(img_w, img_h, ROI, padding)
        
        image_tensor = TF.crop(image_tensor, *cutout)
        image_tensor = TF.resize(image_tensor, [128, 128])
        return image_tensor
    
    def _calculate_roi_cutout(self, img_h, img_w, ROI, padding):
        top = max(0, ROI['top'] - padding)
        left = max(0, ROI['left'] - padding)
        bottom = min(img_h, ROI['bottom'] + padding)
        right = min(img_w, ROI['right'] + padding)

        height = bottom - top
        width = right - left
        
        return top, left, height, width
        
    def _load_pickles(self):
        for dump_file in os.listdir(self.path):
            path = pjoin(self.path, dump_file)
            with open(path, 'rb') as dump:
                patient = pickle.load(dump)
                self.data.append(patient)
    
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
    def __init__(self, path_to_pickles):
        self.path = pjoin(path_to_pickles, 'sale')
        self.num_frames_per_sequence = None
        self.data = []
        
        self._load_pickles()
    
    def _transform(self, image_tensor):
        image_tensor = TF.resize(image_tensor, [128, 128])
        return image_tensor
    
    def _load_pickles(self):
        for dump_file in os.listdir(self.path):
            path = pjoin(self.path, dump_file)
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
            indices = list(range(len(patient.images)))
            while len(indices) > self.num_frames_per_sequence:
                to_remove = random.choice(indices)
                indices.remove(to_remove)
            
            images = [patient.images[i] for i in indices]
        
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
