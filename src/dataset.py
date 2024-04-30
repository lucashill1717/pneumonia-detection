from keras.utils import PyDataset
from math import ceil
from skimage.io import imread
from skimage.transform import resize
import numpy as np

def _get_image_paths(dir):
    pass

class PneumoniaDataset(PyDataset):
    """
    Dataset specifically tailored to the Pneumonia Dataset.
    Utilizes a Keras PyDataset class, allowing a generator
    to be used to avoid memory issues.
    """
    def __init__(self, x, y, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return np.array(
            [resize(imread(file_name), (1416, 1736)) for file_name in batch_x]
        ), np.array(batch_y)
