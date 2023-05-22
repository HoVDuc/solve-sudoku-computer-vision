import numpy as np
import pandas as pd
import cv2


from torch.utils.data import Dataset
import torch
from PIL import Image


class SudokuData(Dataset):

    def __init__(self, path, df, device, transform=None):
        self.df = df.to_numpy()
        self.device = device
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df[index][0]
        folder = self.df[index][1]
        label = torch.tensor(int(folder), device=self.device)
        file_path = f"{self.path}{folder}/{filename}"
        image = cv2.imread(file_path)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            image = self.transform(image)
        image = image.to(self.device)
        label = label.to(self.device)
        return image, label