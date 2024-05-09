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
    Activation,
)


training_data = PneumoniaDataset("data/train", 4)

model = Sequential(
    layers=[
        Input(shape=(HEIGHT, WIDTH, 1)),
        Conv2D(32, 3),
        BatchNormalization(),
        Activation("elu"),
        Conv2D(32, 3),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(2),
        Dropout(0.1),
        Conv2D(64, 3),
        BatchNormalization(),
        Activation("elu"),
        Conv2D(64, 3),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(3),
        Dropout(0.1),
        Conv2D(128, 3),
        BatchNormalization(),
        Activation("elu"),
        Conv2D(128, 3),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.1),
        Flatten(),
        Dense(3, activation="softmax"),
        # global pooling and sigmoid activation?
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.fit(x=training_data, epochs=100)

model.save("model0.keras")

# 816, 1136
