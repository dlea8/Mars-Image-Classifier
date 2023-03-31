import numpy as np
import pandas as pd
import torch.utils.data as data
import torch
import os 
import skimage.io as io
from torchvision.io import read_image


class MarsDataset(data.Dataset):


    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = read_image(img_path)
        label = self.landmarks_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.transform:
            label = self.transform(label)
        return image, label
