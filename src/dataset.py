from keras.utils import PyDataset
from math import ceil

class PneumoniaDataset(PyDataset):
    def __init__(self, x, y, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y = x, y
        self.batch_size = batch_size
    
    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

# Average image height: 1416
# Average image width: 1736
