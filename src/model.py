from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from dataset import PneumoniaDataset


training_data = image_dataset_from_directory(
    directory="data/train",
    image_size=(320, 320),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical",
    verbose=False,
)

model = Sequential(
    layers=[
        Input(shape=(320, 320, 1)),
        Conv2D(64, 3, activation="relu"),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(128, 3, activation="relu"),
        Conv2D(128, 3, activation="relu"),
        Dropout(rate=0.1),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.fit(x=training_data, epochs=100)

model.save("model0.keras")
