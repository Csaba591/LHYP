import numpy as np
from dicom_reader import DCMreaderVM
from sale_reader import SaleReader
from con2img import draw_contourmtcs2image as draw
from os.path import join as pjoin
import os
import sys
import pickle
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom import dcmread

from dicom_utils import *

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Patient:
    def __init__(self, images, patient_data, pathology):
        self.images = images # numpy array
        self.patient_data = patient_data
        self.label = pathology 

AXIS = 'sa'

def process_image(image, dataset):
    rescaled = apply_modality_lut(image, dataset)
    perc_cut = percentile_cut(rescaled)
    return normalize(perc_cut)

def get_images_sa(dr):
    slice_skip = dr.num_slices // 3
    frame_skip = 3
    images = []
    for s in range(0, dr.num_slices, slice_skip):
        for f in range(0, dr.num_frames, frame_skip):
            image = dr.get_image(s, f)
            ds = dcmread(dr.get_dcm_path(s, f))
            processed = process_image(image, ds)
            images.append(processed)
    return images

def get_images_sale(dr):
    images = []
    type_index = 0 # take a single sequence only
    for seq_index in range(dr.num_frames_per_sequence):
        ds = dcmread(dr.get_path(type_index, seq_index))
        image = dr.get_image(type_index, seq_index)
        processed = process_image(image, ds)
        images.append(processed)
    return images

def generate_patient_dataset(patient_path):
    axes = [d for d in os.listdir(patient_path) if os.path.isdir(pjoin(patient_path, d))]
    if AXIS not in axes:
        eprint('Invalid axis', AXIS)
        return None
    
    axis_path = pjoin(patient_path, AXIS)
    if AXIS == 'sa':
        images_path = pjoin(axis_path, 'images')
        dr = DCMreaderVM(images_path)
        image_reader_fn = get_images_sa
    elif AXIS == 'sale':
        dr = SaleReader(axis_path)
        image_reader_fn = get_images_sale
    else:
        eprint(r"Axis was not 'sa' nor 'sale'")
        return None
    
    if dr.broken or dr.num_images == 0:
        return None
    
    images = image_reader_fn(dr)
    
    patient_data = dr.patient_data
    label = get_label(patient_path)
    
    return Patient(images, patient_data, label)

def dump_data(path, patient_id, data):
    eprint('Dumping data for', patient_id)
    filename = pjoin(path, patient_id + '.pickle')
    with open(filename, 'wb') as dump_file:
        pickle.dump(data, dump_file, pickle.HIGHEST_PROTOCOL)
    eprint('Dumped to', filename)

def generate_dataset(samples_path, out_path):
    patient_ids = [ID for ID in os.listdir(samples_path) if os.path.isdir(pjoin(samples_path, ID))]
    for patient_id in patient_ids:
        patient_path = pjoin(samples_path, patient_id)
    
        eprint('Started for patient', patient_id)
        data = generate_patient_dataset(patient_path)
        if data is None:
            eprint('Failed! No images were found.')
            continue
            
        pickled_path = pjoin('..', 'pickled_samples')
        if not os.path.exists(pickled_path):
            os.mkdir(pickled_path)
        
        
        dump_data(pickled_path, patient_id, data)

if __name__ == '__main__':
    path_to_samples = pjoin('..', 'samples')
    out_path = pjoin('..', 'pickled_samples')
    generate_dataset(path_to_samples, out_path)