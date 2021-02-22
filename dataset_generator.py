import numpy as np
from con_reader import CONreaderVM
from dicom_reader import DCMreaderVM
from con2img import draw_contourmtcs2image as draw
from os.path import join as pjoin
import os
import sys
import pickle

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def grayscale(image):
    img_mtx = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(float)
    p1, p99 = np.percentile(img_mtx, (1, 99))
    img_mtx[img_mtx < p1] = p1
    img_mtx[img_mtx > p99] = p99
    img_mtx = (img_mtx - p1) / (p99 - p1)
    return img_mtx

def get_label(patient_folder):
    with open(pjoin(patient_folder, 'meta.txt'), 'rt') as meta:
        label = meta.readline().split(': ')[1].strip()
    return label

AXIS = 'lale'

def generate_patient_dataset(patient_folder):
    patient_dirs = [d for d in os.listdir(patient_folder) if os.path.isdir(pjoin(patient_folder, d))]
    if AXIS not in patient_dirs:
        
        return None
    
    axis_path = pjoin(patient_folder, AXIS)
    if AXIS == 'sa':
        axis_path = pjoin(axis_path, 'images')
    
    dr = DCMreaderVM(axis_path)
    if (dr.num_frames * dr.num_slices == 0 or dr.num_images == 0):
        return None
    
    images = []
    for s in range(dr.num_slices):
        for f in range(dr.num_frames):
            image = dr.get_image(s, f)
            #eprint(image.shape)
            gray = grayscale(image)
            images.append(gray)
            
    label = get_label(patient_folder)
    
    return { 'images': images, 'label': label }

def dump_data(path, patient_id, data):
    eprint('Dumping data...')
    #eprint('data:')
    #eprint(data)
    filename = pjoin(path, patient_id + '.pickle')
    with open(filename, 'wb') as dump_file:
        pickle.dump(data, dump_file, pickle.HIGHEST_PROTOCOL)
    eprint('Dumped to ', patient_id + '.pickle')

if __name__ == '__main__':
    path_to_samples = pjoin('..', 'samples')
    patient_ids = [ID for ID in os.listdir(path_to_samples) if os.path.isdir(pjoin(path_to_samples, ID))]
    for patient_id in patient_ids:
        patient_path = pjoin(path_to_samples, patient_id)
    
        data = generate_patient_dataset(patient_path)
        if data is None:
            # Nano-n STDERR-re!
            eprint('Failed for patient', patient_id)
            continue
            
        pickled_path = pjoin('..', 'pickled_samples')
        if not os.path.exists(pickled_path):
            os.mkdir(pickled_path)
        
        dump_data(pickled_path, patient_id, data)