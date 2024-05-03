from typing import Any
from keras.utils import PyDataset
from math import ceil
from skimage.io import imread
from skimage.transform import resize
from numpy import array, ndarray, eye
from os import listdir


HEIGHT = 512
WIDTH = 512


def get_image_paths(dir: str) -> tuple[list[str], ndarray]:
    paths, classes = [], []
    class_labels = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}
    num_classes = len(class_labels)

    for label, class_idx in class_labels.items():
        label_paths = [f"{dir}/{label}/{file}" for file in listdir(f"{dir}/{label}")]
        paths.extend(label_paths)
        classes.extend([class_idx] * len(label_paths))

    one_hot_labels = eye(num_classes)[classes]
    return paths, one_hot_labels


class PneumoniaDataset(PyDataset):
    """
    Dataset class specifically tailored to the Pneumonia Detection dataset from Kaggle.
    Utilizes a Keras PyDataset class, allowing a generator to be used to avoid memory issues.
    """

    def __init__(
        self: "PneumoniaDataset", dir: str, batch_size: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.x, self.y = get_image_paths(dir)
        self.batch_size = batch_size

    def __len__(self: "PneumoniaDataset") -> int:
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self: "PneumoniaDataset", idx: int) -> tuple[ndarray, ndarray]:
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]
        bad_output = [
            resize(imread(file_name), (HEIGHT, WIDTH)) for file_name in batch_x
        ]
        good_output = []
        for output in bad_output:
            new_output = (
                output.mean(axis=2) if output.shape == (HEIGHT, WIDTH, 3) else output
            )
            good_output.append(new_output)
        return array(good_output), batch_y
