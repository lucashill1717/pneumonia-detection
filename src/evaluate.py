from keras.preprocessing import image_dataset_from_directory
from keras.saving import load_model

model = load_model("model0.keras")

validation_data = image_dataset_from_directory(
    directory="data/val",
    image_size=(320, 320),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical",
    verbose=False,
)

model.evaluate(validation_data)  # type: ignore
