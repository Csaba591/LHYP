from argparse import ArgumentParser
import os
from PIL import Image
import numpy as np
from dicom_reader import DCMreaderVM
from sale_reader import SaleReader
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom import dcmread

parser = ArgumentParser()

parser.add_argument('dicoms_path', type=str)
parser.add_argument('--offset', dest='offset', default=-1, type=int)
parser.add_argument('--fps', dest='fps', default=15, type=int)

args = parser.parse_args()

dicoms_path = args.dicoms_path
offset = args.offset
fps = args.fps

duration = 1000 // fps

def process(img_mtx, dataset):
    img_mtx = apply_modality_lut(img_mtx, dataset)
    p1, p99 = np.percentile(img_mtx, (1, 99))
    img_mtx[img_mtx < p1] = p1
    img_mtx[img_mtx > p99] = p99
    img_mtx = (img_mtx - p1) / (p99 - p1)
    return img_mtx

def get_sa_images(reader):
    if reader.broken:
        return []
    
    if offset != -1:
        frames = [
            Image.fromarray(process(reader.get_image(offset, i), reader)) 
            for i in range(reader.num_frames)]
    else:
        frames = []
        for slice in range(reader.num_slices):
            for frame in range(reader.num_frames):
                ds = dcmread(reader.get_dcm_path(slice, frame))
                frames.append(Image.fromarray(process(reader.get_image(slice, frame), ds)))

def get_sale_images(dr):
    images = []
    if dr.broken: return images
    
    for seq_index in range(dr.num_sequences):
        for frame_index in range(dr.num_frames_per_sequence):
            ds = dcmread(dr.get_path(seq_index, frame_index))
            image = dr.get_image(seq_index, frame_index)
            image = process(image, ds)
            image = ((image - image.min()) * (1/(image.max() - image.min()) * 255)).astype('uint8')
            images.append(Image.fromarray(image))
    return images

axis = 'sa'
if 'sale' in dicoms_path: axis = 'sale'

reader = DCMreaderVM(dicoms_path) if axis == 'sa' else SaleReader(dicoms_path)

print('Reading frames from dicom')

os.makedirs('dicom_gifs', exist_ok=True)
save_path = os.path.join('dicom_gifs', f"{dicoms_path}.gif")

frames = get_sa_images(reader) if axis == 'sa' else get_sale_images(reader)

print(f'Saving to {save_path}')
frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
