from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

DIMENSION = 256

training_data = image_dataset_from_directory(
    directory="data/train",
    image_size=(256, 256),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical",
    verbose=False,
)

model = Sequential(
    layers=[
        Input(shape=(DIMENSION, DIMENSION, 1)),
        Conv2D(64, 3, activation="relu"),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(128, 3, activation="silu"),
        Conv2D(128, 3, activation="silu"),
        MaxPooling2D(),
        Flatten(),
        Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.fit(
    x=training_data,
    epochs=50,
)

model.save("model0.keras")
