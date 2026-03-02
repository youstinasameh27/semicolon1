import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load model
model = joblib.load("best_headpose_model.pkl")


n_features = model.estimators_[0].n_features_in_


initial_type = [('input', FloatTensorType([None, n_features]))]


onnx_model = convert_sklearn(model, initial_types=initial_type)


with open("headpose_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved successfully")