import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import pydicom as dicom
from PIL import Image

import sys
import numpy as np

import warnings
warnings.filterwarnings("error")

np.set_printoptions(threshold=sys.maxsize)

class LiverDataset(Dataset):
    def __init__(self, csv_labels, csv_labels_type, root_dir, transform=None):
        self.labels = pd.read_csv(csv_labels)
        self.labels_type = pd.read_csv(csv_labels_type)
        self.root_dir = root_dir
        self.transform = transform
        
        self.data = []

        for dir1 in os.listdir(root_dir):
            for dir2 in os.listdir(os.path.join(root_dir, dir1)):
                for dir3 in os.listdir(os.path.join(root_dir, dir1, dir2)):
                    self.data.append([root_dir, dir1, dir2, dir3])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(*self.data[index])
        img = dicom.dcmread(img_path)
        ndarray_image = img.pixel_array.astype(float)
        
        ndarray_image = (np.maximum(ndarray_image, 0)/ndarray_image.max())
        
        img = Image.fromarray(ndarray_image)

        # y_label = torch.zeros(17)
        y_letter = self.labels.loc[(self.labels['DLDS'] == int(self.data[index][1])) & (self.labels['Series'] == int(self.data[index][2]))]['Label'].values[0]
        y_label = self.labels_type.loc[self.labels_type['Label'] == y_letter].index[0]
        # y_label[y_index] = 1

        
        if self.transform:
            img = self.transform(img)
            
        return img, y_label