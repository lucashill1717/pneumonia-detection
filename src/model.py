from keras.preprocessing import image_dataset_from_directory as idfd
from keras import Sequential

training_data = idfd(
    "data/train",
    image_size=(512, 512),
    color_mode="grayscale",
    interpolation="gaussian",
)


