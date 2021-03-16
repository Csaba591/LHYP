from os.path import join as pjoin
import numpy as np
from math import floor, ceil

def percentile_cut(img_mtx):
    p1, p99 = np.percentile(img_mtx, (1, 99))
    img_mtx[img_mtx < p1] = p1
    img_mtx[img_mtx > p99] = p99
    img_mtx = (img_mtx - p1) / (p99 - p1)
    return img_mtx

def rescale(image):
    return ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')

def get_label(patient_folder):
    with open(pjoin(patient_folder, 'meta.txt'), 'rt') as meta:
        label = meta.readline().split(': ')[1].strip()
    return label

def get_patient_data(ds):
    meta_data = {
        'gender': ds.PatientSex,
        'weight': ds.PatientWeight,
        'pregnant': ds.PregnancyStatus,
        'heart_rate': ds.HeartRate,
        'time_of_day': ds.StudyTime
    }
    return meta_data

def get_roi_from_contours(contours):
    top, bottom, left, right = None, None, None, None
    for slice in contours:
        for frame in contours[slice]:
            for mode in contours[slice][frame]:
                for x, y in contours[slice][frame][mode]:
                    if top is None or y < top:
                        top = y
                    if bottom is None or y > bottom:
                        bottom = y
                    if left is None or x < left:
                        left = x
                    if right is None or x > right:
                        right = x
    
    top, bottom, left, right = floor(top), ceil(bottom), floor(left), ceil(right)
    return top, bottom, left, right

# for post processing
def cut_out_roi(ROI, image):
    top = ROI['top']
    bottom = ROI['bottom']
    left = ROI['left']
    right = ROI['right']
    
    roi_w, roi_h = right - left, bottom - top
    masked_w, masked_h = 128, 128
    padding_w, padding_h = (masked_w - roi_w) // 2, (masked_h - roi_h) // 2
    if padding_w < 0 or padding_h < 0:
        eprint('ERROR: sale ROI larger than', (masked_w, masked_h))
    
    # cut out
    cut_out = image[top:bottom, left:right]
    masked = np.zeros((masked_h, masked_w), dtype=np.uint8)
    # place in middle of black image
    masked[padding_h:padding_h+roi_h, padding_w:padding_w+roi_w] = cut_out[:, :]
    return masked
