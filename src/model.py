import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

DIMENSION = 256

training_data = image_dataset_from_directory(
    directory="data/train",
    image_size=(DIMENSION, DIMENSION),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical",
)

model = Sequential(
    layers=(
        Input(shape=(DIMENSION, DIMENSION, 1)),
        Conv2D(64, 3, activation="relu"),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(128, 3, activation="silu"),
        Conv2D(128, 3, activation="silu"),
        MaxPooling2D(),
        Flatten(),
        Dense(3, activation="softmax"),
    )
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.fit(
    x=training_data,
    batch_size=32,
    epochs=20,
    verbose=2,
)
