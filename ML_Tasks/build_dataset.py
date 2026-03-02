import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.io import loadmat
from tqdm import tqdm

# ==============================
# PATH TO DATASET
# ==============================
DATASET_PATH = r"C:\Users\youss\Downloads\archive (1)\AFLW2000"

# ==============================
# Mediapipe Initialization
# ==============================
base_options = mp.tasks.BaseOptions(
    model_asset_path=r"C:\Users\youss\Downloads\AI-Services-add-headpose-project\AI-Services-add-headpose-project\Api\assets\face_landmarker.task"
)

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
SELECTED_LANDMARKS = [
    33, 133, 160, 158, 144,      # left eye
    263, 362, 385, 387, 373,     # right eye
    1, 2, 4, 5,                  # nose
    61, 291, 17,                 # mouth
    152,                         # chin
    234, 454,                    # cheeks
    10                           # forehead
]

data_rows = []

for file in tqdm(os.listdir(DATASET_PATH)):

    if not file.endswith(".jpg"):
        continue

    image_path = os.path.join(DATASET_PATH, file)
    mat_path = image_path.replace(".jpg", ".mat")

    if not os.path.exists(mat_path):
        continue

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        continue

    landmarks = result.face_landmarks[0]
    landmark_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]

    # normalization reference
    left_eye = landmark_list[33]
    right_eye = landmark_list[263]
    left_outer = landmark_list[130]
    right_outer = landmark_list[359]

    eye_center_x = (left_eye["x"] + right_eye["x"]) / 2
    eye_center_y = (left_eye["y"] + right_eye["y"]) / 2

    face_width = np.hypot(
        right_outer["x"] - left_outer["x"],
        right_outer["y"] - left_outer["y"]
    )

    features = []

    for idx in SELECTED_LANDMARKS:
        x = (landmark_list[idx]["x"] - eye_center_x) / face_width
        y = (landmark_list[idx]["y"] - eye_center_y) / face_width
        z = landmark_list[idx]["z"] / face_width
        features.extend([x, y, z])

    mat = loadmat(mat_path)
    pose_para = mat["Pose_Para"][0]

    pitch = pose_para[0]
    yaw   = pose_para[1]
    roll  = pose_para[2]

    features.extend([yaw, pitch, roll])
    data_rows.append(features)

columns = []

for idx in SELECTED_LANDMARKS:
    columns.extend([f"l{idx}_x", f"l{idx}_y", f"l{idx}_z"])

columns.extend(["GT_yaw", "GT_pitch", "GT_roll"])

df = pd.DataFrame(data_rows, columns=columns)
df.to_csv("dataset.csv", index=False)

print("Dataset saved successfully.")