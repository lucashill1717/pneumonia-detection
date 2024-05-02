from dataset import PneumoniaDataset
from keras.saving import load_model

model = load_model("model0.keras")

validation_data = PneumoniaDataset("data/train", 32)

model.evaluate(validation_data)  # type: ignore
