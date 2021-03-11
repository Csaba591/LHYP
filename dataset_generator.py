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

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Patient:
    def __init__(self, patient_id, images, patient_data, pathology):
        self.id = patient_id
        self.images = images # numpy array
        self.patient_data = patient_data
        self.label = pathology

def process_image(image, dataset):
    preproc = apply_modality_lut(image, dataset)
    perc_cut = percentile_cut(preproc)
    return rescale(perc_cut)

def get_images_sa(dr, cr):
    images = []
    if dr.broken: return images
    
    slice_skip = dr.num_slices // 3
    if slice_skip == 0: slice_skip = 1
    
    frames_to_take = [
        0, 1,                                   # diastole
        dr.num_frames//2, dr.num_frames//2 + 1, # inbetween
        -2, -1                                  # systole
    ]
    all_contours = cr.get_hierarchical_contours()
    top, bottom, left, right = get_roi_from_contours(all_contours)
    roi_w, roi_h = right - left, bottom - top
    masked_w, masked_h = 128, 128
    padding_w, padding_h = (masked_w - roi_w) // 2, (masked_h - roi_h) // 2
    
    if padding_w < 0 or padding_h < 0:
        eprint('ERROR: sale ROI larger than', (masked_w, masked_h))
    
    for s in range(0, dr.num_slices, slice_skip):
        for f in frames_to_take:
            image = dr.get_image(s, f)
            ds = dcmread(dr.get_dcm_path(s, f))
            processed = process_image(image, ds)
            # cut out
            roi = processed[top:bottom, left:right]
            masked = np.zeros((masked_h, masked_w), dtype=np.uint8)
            # place in middle of black image
            masked[padding_h:padding_h+roi_h, padding_w:padding_w+roi_w] = roi[:, :]
            images.append(masked)
    return images

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

def generate_patient_dataset(patient_id, patient_path, axis='sa'):
    axes = [d for d in os.listdir(patient_path) if os.path.isdir(pjoin(patient_path, d))]
    if axis not in axes:
        eprint('Patient', patient_id, 'has no data for', axis)
        return None
    
    axis_path = pjoin(patient_path, axis)
    if axis == 'sa':
        images_path = pjoin(axis_path, 'images')
        dr = DCMreaderVM(images_path)
        cr = CONreaderVM(pjoin(axis_path, 'contours.con'))
        images = get_images_sa(dr, cr)
    elif axis == 'sale':
        dr = SaleReader(axis_path)
        images = get_images_sale(dr)
    else:
        eprint(r"Axis was not 'sa' nor 'sale'")
        return None
    
    if len(images) == 0:
        return None

    patient_data = dr.patient_data
    label = get_label(patient_path)
    
    return Patient(patient_id, images, patient_data, label)

def dump_data(path, patient_id, data):
    eprint('Dumping data for', patient_id)
    filename = pjoin(path, patient_id + '.pickle')
    with open(filename, 'wb') as dump_file:
        pickle.dump(data, dump_file, pickle.HIGHEST_PROTOCOL)
    eprint('Dumped to', filename)

def generate_dataset(samples_path, out_path, axis='sa'):
    patient_ids = [ID for ID in os.listdir(samples_path) if os.path.isdir(pjoin(samples_path, ID))]
    for patient_id in patient_ids:
        patient_path = pjoin(samples_path, patient_id)
    
        eprint('Started for patient', patient_id)
        data = generate_patient_dataset(patient_id, patient_path, axis=axis)
        if data is None:
            eprint('Failed! No images were found.')
            continue
            
        os.makedirs(out_path, exist_ok=True)
        
        dump_data(out_path, patient_id, data)

if __name__ == '__main__':
    path_to_samples = pjoin('..', 'samples')
    for axis in ['sa', 'sale']:
        eprint('--- Generating', axis, 'data')
        out_path = pjoin('..', 'pickled_samples', axis)
        generate_dataset(path_to_samples, out_path, axis=axis)
        eprint('---', axis, 'done')
        