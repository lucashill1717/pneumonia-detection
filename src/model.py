from dataset import PneumoniaDataset, HEIGHT, WIDTH
from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)


training_data = PneumoniaDataset("data/train", 4)

model = Sequential(
    layers=[
        Input(shape=(HEIGHT, WIDTH, 1)),
        Conv2D(16, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, 3, padding="same", activation="relu"),
        Dropout(0.1),
        BatchNormalization(),
        MaxPooling2D(2),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(128, 3, padding="same", activation="relu"),
        Dropout(0.1),
        BatchNormalization(),
        MaxPooling2D(2),
        Conv2D(256, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),
        Conv2D(512, 3, padding="same", activation="relu"),
        Dropout(0.1),
        BatchNormalization(),
        MaxPooling2D(2),
        Flatten(),
        Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
# model.summary();exit()
model.fit(x=training_data, epochs=100)

model.save("model1.keras")
