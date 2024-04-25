from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense

training_data = image_dataset_from_directory(
    directory="data/train",
    image_size=(512, 512),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical"
)

model = Sequential(layers=(
    Input(shape=(512, 512, 1)),
    Conv2D(64, 3, activation="relu"),
    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),
    Conv2D(128, 3, activation="silu"),
    Conv2D(128, 3, activation="silu"),
    MaxPooling2D(),
    Dense(3, activation="softmax")
))

model.compile(
    optimizer="adam",
    loss="categoricalcrossentropy",
    metrics=(
        
    )
)
