from dataset import PneumoniaDataset
from keras.saving import load_model

model = load_model("model2.keras")

test_data = PneumoniaDataset("data/test", 4)

model.evaluate(test_data)  # type: ignore
