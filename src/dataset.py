from typing import Any
from keras.utils import PyDataset
from math import ceil
from skimage.io import imread
from skimage.transform import resize
from numpy import array, ndarray


def _get_image_paths(dir: str) -> tuple[list[str], list[list]]:
    paths, classes = [], []
    return paths, classes


class PneumoniaDataset(PyDataset):
    """
    Dataset class specifically tailored to the Pneumonia Detection dataset from Kaggle.
    Utilizes a Keras PyDataset class, allowing a generator to be used to avoid memory issues.
    """

    def __init__(
        self: "PneumoniaDataset", dir: str, batch_size: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.x, self.y = _get_image_paths(dir)
        self.batch_size = batch_size

    def __len__(self: "PneumoniaDataset") -> int:
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self: "PneumoniaDataset", idx: int) -> tuple[ndarray, ndarray]:
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        return array(
            [resize(imread(file_name), (1416, 1736)) for file_name in batch_x]
        ), array(batch_y)
