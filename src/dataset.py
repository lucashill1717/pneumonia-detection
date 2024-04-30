from keras.utils import PyDataset
from math import ceil
from skimage.io import imread
from skimage.transform import resize
import numpy as np

def _get_image_paths(dir: str) -> tuple[list[str], list[list]]:
    paths, classes = [], []
    return paths, classes

class PneumoniaDataset(PyDataset):
    """
    Dataset specifically tailored to the Pneumonia Dataset.
    Utilizes a Keras PyDataset class, allowing a generator
    to be used to avoid memory issues.
    """
    def __init__(self, dir: str, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = _get_image_paths(dir)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx: int):
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return np.array(
            [resize(imread(file_name), (1416, 1736)) for file_name in batch_x]
        ), np.array(batch_y)
