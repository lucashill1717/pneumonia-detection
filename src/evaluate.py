from keras.preprocessing import image_dataset_from_directory
from keras.saving import load_model

model = load_model("model0.keras")

validation_data = image_dataset_from_directory(
    directory="data/val",
    image_size=(256, 256),
    color_mode="grayscale",
    interpolation="gaussian",
    label_mode="categorical",
    verbose=False,
)

loss, accuracy = model.evaluate(validation_data) # type: ignore
print("Validation Accuracy:", accuracy)
