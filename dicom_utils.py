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
        'gender': ds['0x00100040'].value,
        'weight': ds['0x00101030'].value,
        'pregnant': ds['0x001021C0'].value,
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
    margin = 2
    if top >= margin: top -= margin
    bottom += margin
    if left >= margin: left -= margin
    right += margin
    return top, bottom, left, right
