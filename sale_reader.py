import pydicom as dicom
from os.path import join as pjoin
import os
import sys
import numpy as np
from dicom_utils import get_patient_data

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class SaleReader:
    def __init__(self, path):
        self.path = path
        self.images = []
        self.num_images = 0
        self.num_image_types = 0
        self.dcm_paths = []
        self.patient_data = {}
        self.broken = False
        self._read_images()
        
    def _read_images(self):
        files = [f for f in os.listdir(self.path) if f.lower().endswith('.dcm')]
        if len(files) == 0:
            return
        
        # data: {
        #     'SeriesDescription': {
        #         'ImageType': [(pixel_array, SliceLocation, path), ...],
        #         ...
        #     },
        #     ...   
        # }
        slice_sequences_by_imagetype_by_seriesdesc = {}
        for f in files:
            path = pjoin(self.path, f)
            try:
                ds = dicom.dcmread(path)
                series_desc = ''.join(ds.SeriesDescription)
                if series_desc not in slice_sequences_by_imagetype_by_seriesdesc:
                    slice_sequences_by_imagetype_by_seriesdesc[series_desc] = {}
                
                image_type = '-'.join(ds.ImageType)
                if image_type not in slice_sequences_by_imagetype_by_seriesdesc[series_desc]:
                    slice_sequences_by_imagetype_by_seriesdesc[series_desc][image_type] = []
                    self.num_image_types += 1
                
                slice_sequences_by_imagetype_by_seriesdesc[series_desc][image_type].append((ds.pixel_array, ds.SliceLocation, path))
                self.num_images += 1
            except:
                eprint(f, 'is broken.')
                self.broken = True
                return
        
        self.patient_data = get_patient_data(ds)
        
        eprint(slice_sequences_by_imagetype_by_seriesdesc.keys())
        for key in slice_sequences_by_imagetype_by_seriesdesc:
            print(slice_sequences_by_imagetype_by_seriesdesc[key].keys())
        for series in slice_sequences_by_imagetype_by_seriesdesc:
            for imagetype in slice_sequences_by_imagetype_by_seriesdesc[series]:
                slice_sequences_by_imagetype_by_seriesdesc[series][imagetype] = \
                    sorted(
                        slice_sequences_by_imagetype_by_seriesdesc[series][imagetype], 
                        key=lambda e: e[1], reverse=True)
                for i in list(slice_sequences_by_imagetype_by_seriesdesc[series].values()):
                    eprint(len(i), series, imagetype)
        
        size_h, size_w = slice_sequences_by_imagetype_by_seriesdesc[series][imagetype][0][0].shape
        
        self.num_frames_per_sequence = len(list(slice_sequences_by_imagetype_by_seriesdesc[series].values())[0])
        
        self.images = np.ones((self.num_image_types, self.num_frames_per_sequence, size_h, size_w))
        self.dcm_paths = np.zeros((self.num_image_types, self.num_frames_per_sequence), dtype=object)
        
        type_index = 0
        for series in slice_sequences_by_imagetype_by_seriesdesc:
            for imagetype in slice_sequences_by_imagetype_by_seriesdesc[series]:
                for frame_index, (frame, SliceLocation, dcm_path) in enumerate(slice_sequences_by_imagetype_by_seriesdesc[series][imagetype]):
                    self.dcm_paths[type_index, frame_index] = dcm_path
                    self.images[type_index, frame_index, :, :] = frame
                type_index += 1
        print(self.dcm_paths)
    
    def get_image(self, image_type, slice):
        return self.images[image_type, slice]
    
    def get_path(self, image_type, slice):
        return self.dcm_paths[image_type, slice]