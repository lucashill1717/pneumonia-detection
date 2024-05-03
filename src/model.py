from dataset import PneumoniaDataset
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


training_data = PneumoniaDataset("data/train", 8)

model = Sequential(
    layers=[
        Input(shape=(816, 1136, 1)),
        Conv2D(32, 3, activation="relu"),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(pool_size=(3, 3)),
        Conv2D(64, 3, activation="relu"),
        Conv2D(64, 3, activation="relu"),
        Dropout(rate=0.1),
        MaxPooling2D(pool_size=(3, 3)),
        Flatten(),
        Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.fit(x=training_data, epochs=50)

model.save("model0.keras")
