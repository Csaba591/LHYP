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
from con_reader import CONreaderVM
from dicom_utils import *
from utils import get_logger, progress_bar
import time

logger = get_logger(__name__)

class Patient:
    def __init__(self, 
        patient_id, axis, 
        images, patient_data, 
        pathology, SA_ROI=None):
        
        self.id = patient_id
        self.axis = axis
        self.images = images # numpy array
        self.patient_data = patient_data
        self.label = pathology
        self.ROI = SA_ROI

def process_image(image, dataset):
    preproc = apply_modality_lut(image, dataset)
    perc_cut = percentile_cut(preproc)
    return rescale(perc_cut)

def get_images_sa(dr, cr):
    images = []
    ROI = {}
    if dr.broken: return images, ROI
    
    slice_skip = dr.num_slices // 3
    if slice_skip == 0: slice_skip = 1
    
    frames_to_take = [
        0, 1,                                   # diastole
        dr.num_frames//2, dr.num_frames//2 + 1, # inbetween
        -2, -1                                  # systole
    ]
    all_contours = cr.get_hierarchical_contours()
    top, bottom, left, right = get_roi_from_contours(all_contours)
    
    ROI['top'] = top
    ROI['bottom'] = bottom
    ROI['left'] = left
    ROI['right'] = right
    
    for s in range(0, dr.num_slices, slice_skip):
        for f in frames_to_take:
            image = dr.get_image(s, f)
            ds = dcmread(dr.get_dcm_path(s, f))
            processed = process_image(image, ds)
            images.append(processed)
    return images, ROI

def get_images_sale(dr):
    images = []
    if dr.broken: return images
    
    seq_index = 0 # take a single sequence only
    for frame_index in range(dr.num_frames_per_sequence):
        ds = dcmread(dr.get_path(seq_index, frame_index))
        image = dr.get_image(seq_index, frame_index)
        processed = process_image(image, ds)
        images.append(processed)
    return images

def generate_patient_dataset(patient_id, patient_path, axis):
    axes = [d for d in os.listdir(patient_path) if os.path.isdir(pjoin(patient_path, d))]
    if axis not in axes:
        logger.warning(f'Patient {patient_id} has no data for {axis}')
        return None
    
    axis_path = pjoin(patient_path, axis)
    if axis == 'sa':
        images_path = pjoin(axis_path, 'images')
        dr = DCMreaderVM(images_path)
        cr = CONreaderVM(pjoin(axis_path, 'contours.con'))
        images, ROI = get_images_sa(dr, cr)
    elif axis == 'sale':
        dr = SaleReader(axis_path)
        images = get_images_sale(dr)
        ROI = None
    else:
        logger.error(r"Axis was not 'sa' nor 'sale'")
        return None
    
    if len(images) == 0:
        return None

    patient_data = dr.patient_data
    label = get_label(patient_path)
    
    return Patient(patient_id, axis, images, patient_data, label, SA_ROI=ROI)

def dump_data(path, patient_id, data):
    filename = pjoin(path, patient_id + '.pickle')
    with open(filename, 'wb') as dump_file:
        pickle.dump(data, dump_file, pickle.HIGHEST_PROTOCOL)

def generate_dataset(samples_path, out_path):
    patient_ids = [ID for ID in os.listdir(samples_path) if os.path.isdir(pjoin(samples_path, ID))]
    
    axes = ['sa', 'sale']
    for axis_index, axis in enumerate(axes):
        for pat_index, patient_id in enumerate(patient_ids):
            patient_path = pjoin(samples_path, patient_id)
        
            data = generate_patient_dataset(patient_id, patient_path, axis=axis)
            if data is None: continue
            
            axis_path = pjoin(out_path, axis)
            os.makedirs(axis_path, exist_ok=True)
            
            dump_data(axis_path, patient_id, data)
            
            progress_bar(
                axis_index*len(patient_ids) + pat_index + 1, 
                len(axes)*len(patient_ids), 
                20)

if __name__ == '__main__':
    path_to_samples = pjoin('..', 'samples')
    out_path = pjoin('..', 'pickled_samples')
    
    generate_dataset(path_to_samples, out_path)
        