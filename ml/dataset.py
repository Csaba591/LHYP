from os.path import join as pjoin
import os
import sys
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import random

# go up one level
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset_generator import Patient  

def get_label_binary(pathology) -> int:
    label = pathology.lower()
    if 'u18' in label or 'normal' in label or 'sport' in label:
        return 0
    return 1

avg_std = np.mean([0.229, 0.224, 0.225])
avg_mean = np.mean([0.485, 0.456, 0.406])

class SADataset(Dataset):
    def __init__(self, path_to_pickles, patient_ids, for_training=True, image_size=(128, 128)):
        self.path = pjoin(path_to_pickles, 'sa')
        self.data = []
        self.num_frames_per_slice = 6
        self.should_augment = for_training
        self.im_size = image_size

        self.transforms = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.15)
        ])
        
        self._load_pickles(patient_ids, for_training)
    
    def _transform(self, image_tensor, ROI, padding=2):
        img_h, img_w = image_tensor.shape[-2], image_tensor.shape[-1]
        cutout = self._calculate_roi_cutout(img_w, img_h, ROI, padding)
        
        image_tensor = TF.crop(image_tensor, *cutout)
        
        if self.should_augment:
            image_tensor = self.transforms(image_tensor)
        
        image_tensor = TF.resize(image_tensor, self.im_size)
        image_tensor = TF.normalize(image_tensor, mean=avg_mean, std=avg_std)
        return image_tensor
    
    def _calculate_roi_cutout(self, img_h, img_w, ROI, padding):
        top = max(0, ROI['top'] - padding)
        left = max(0, ROI['left'] - padding)
        bottom = min(img_h, ROI['bottom'] + padding)
        right = min(img_w, ROI['right'] + padding)

        height = bottom - top
        width = right - left
        
        return top, left, height, width
    
    def print_stats(self):
        print(self.label_stats, file=sys.stderr)
    
    def _load_pickles(self, ids, for_training):
        self.label_stats = {}
        label_counts = [0, 0]
        sample_weights = [0]*len(ids)
        
        for i, ID in enumerate(ids):
            path = pjoin(self.path, ID+'.pickle')
            with open(path, 'rb') as dump:
                patient = pickle.load(dump)
                self.data.append(patient)
                
                if patient.label in self.label_stats:
                    self.label_stats[patient.label] += 1
                else: self.label_stats[patient.label] = 1
                
                label_index = get_label_binary(patient.label)
                label_counts[label_index] += 1
                sample_weights[i] = label_index
        
        self.sampler = None
        if for_training:
            label_weights = [1/x for x in label_counts]
            for i in range(len(sample_weights)):
                sample_weights[i] = label_weights[sample_weights[i]]
                    
            self.sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True)
    
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
        
        label = get_label_binary(patient.label)
        # if label == 0: label = -1
        label_tensor = torch.Tensor([label])
        
        return (transformed_images_tensor, label_tensor)
    
    @property
    def num_channels(self):
        return self.num_frames_per_slice
    
class SALEDataset(Dataset):
    def __init__(self, path_to_pickles, patient_ids, for_training=True, image_size=(128, 128)):
        self.path = pjoin(path_to_pickles, 'sale')
        self.num_frames_per_sequence = None
        self.data = []
        self.should_augment = for_training
        self.im_size = image_size
        
        self.transforms = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.15)
        ])
        
        self._load_pickles(patient_ids, for_training)
    
    def _transform(self, image):
        image_tensor = TF.to_tensor(image)
        
        if self.should_augment:
            crop = transforms.RandomCrop(int(image_tensor.shape[-1]*0.8))
            image_tensor = crop(image_tensor)
            image_tensor = self.transforms(image_tensor)

        image_tensor = TF.resize(image_tensor, self.im_size)
        image_tensor = TF.normalize(image_tensor, mean=avg_mean, std=avg_std)
        return image_tensor
    
    def _load_pickles(self, ids, for_training):
        self.label_stats = {}
        label_counts = [0, 0]
        sample_weights = [0]*len(ids)
        
        for i, ID in enumerate(ids):
            path = pjoin(self.path, ID+'.pickle')
            with open(path, 'rb') as dump:
                patient = pickle.load(dump)
                self.data.append(patient)
                
                if patient.label in self.label_stats:
                    self.label_stats[patient.label] += 1
                else: self.label_stats[patient.label] = 1
                
                if self.num_frames_per_sequence is None or len(patient.images) < self.num_frames_per_sequence:
                    self.num_frames_per_sequence = len(patient.images)
                    
                label_index = get_label_binary(patient.label)
                label_counts[label_index] += 1
                sample_weights[i] = label_index
    
        self.sampler = None
        if for_training:
            label_weights = [1/x for x in label_counts]
            for i in range(len(sample_weights)):
                sample_weights[i] = label_weights[sample_weights[i]]
    
            self.sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True)
        
    def print_stats(self):
        print(self.label_stats, file=sys.stderr)
    
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
        
        transformed_images = [self._transform(im) for im in images]
        
        transformed_images_tensor = torch.cat(transformed_images)
        
        label_tensor = torch.Tensor([get_label_binary(patient.label)])
        
        return (transformed_images_tensor, label_tensor)
    
    @property
    def num_channels(self):
        return self.num_frames_per_sequence

    @num_channels.setter
    def num_channels(self, x):
        self.num_frames_per_sequence = x
