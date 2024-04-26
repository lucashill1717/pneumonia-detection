from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

training_data = image_dataset_from_directory(
    directory="data/train",
    image_size=(384, 384),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical",
    verbose=False,
)

model = Sequential(
    layers=[
        Input(shape=(384, 384, 1)),
        Conv2D(128, 5, activation="relu"),
        Conv2D(128, 3, activation="relu"),
        Dropout(rate=0.1),
        MaxPooling2D(),
        Conv2D(256, 5, activation="relu"),
        Conv2D(256, 3, activation="relu"),
        Dropout(rate=0.1),
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
    epochs=100,
)

model.save("model0.keras")
