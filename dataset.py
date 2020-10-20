import os
import random
import numpy as np
from PIL import Image
from skimage import io
from skimage.color import gray2rgb


import torch
from torch.utils.data import Dataset

import warnings

class AEI_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Dataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transform = transform


    def __getitem__(self, index):
        l = len(self.files)
        s_idx = index%l
        if index >= 4*l:
            f_idx = s_idx

        else:
            f_idx = random.randrange(l)

        # if f_idx == 0:
        #     f_idx = s_idx
        # else:
        #     f_idx = random.randrange(l)

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files) * 5


class AEI_Val_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Val_Dataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transfrom = transform

    def __getitem__(self, index):
        l = len(self.files)

        f_idx = index // l
        s_idx = index % l

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transfrom is not None:
            f_img = self.transfrom(f_img)
            s_img = self.transfrom(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files) * len(self.files)