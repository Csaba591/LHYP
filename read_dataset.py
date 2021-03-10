from os.path import join as pjoin
import os
from matplotlib import pyplot as plt
import pickle
import numpy as np
from dataset_generator import Patient

if __name__ == '__main__':
    samples_path = pjoin('..', 'pickled_samples')
    # for patient_id in os.listdir(samples_path):
    #     patient_dir = pjoin(samples_path, patient_id)
        
    #     with open(patient_dir, 'rb') as dump_file:
    #         data = pickle.load(dump_file)
    #     print(data.id)
    #     print(data.images[0].shape)
    #     print(data.images[0].min(), data.images[0].max())
    #     print(data.label)
    #     print(data.patient_data)
        
    #     plt.imshow(data.images[0], cmap='gray')
    #     plt.show()
    patient_dir = pjoin(samples_path, '9102110AMR806.pickle')
    with open(patient_dir, 'rb') as dump_file:
        data = pickle.load(dump_file)
        print(data.images[0].shape)
        print(data.images[0].min(), data.images[0].max())
        print(data.label)
        print(data.patient_data)
    for image in data.images:
        plt.imshow(image, cmap='gray')
        plt.show()