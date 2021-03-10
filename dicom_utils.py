from os.path import join as pjoin
import numpy as np

def percentile_cut(img_mtx):
    p1, p99 = np.percentile(img_mtx, (1, 99))
    img_mtx[img_mtx < p1] = p1
    img_mtx[img_mtx > p99] = p99
    img_mtx = (img_mtx - p1) / (p99 - p1)
    return img_mtx

def normalize(image):
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