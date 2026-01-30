import os

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATHS = [
    "51ep-16-GPU.pt",
    os.path.join("runs", "detect", "train5", "weights", "best.pt"),
]


@st.cache_resource
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return YOLO(path), path
    return None, None


def count_classes(results, class_names):
    counts = {name: 0 for name in class_names.values()}
    if results.boxes is None or results.boxes.cls is None:
        return counts
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    for class_id in class_ids:
        name = class_names.get(int(class_id), f"class_{class_id}")
        counts[name] = counts.get(name, 0) + 1
    return counts


st.set_page_config(page_title="UAV Detection", layout="wide")
st.title("UAV Types Detection Web App")
st.write(
    "Upload one or more images to detect and count multirotor and fixed-wing UAVs."
)

model, model_path = load_model()
if model is None:
    st.error("Model file not found. Please place weights in the project folder.")
    st.stop()

st.caption(f"Loaded model: `{model_path}`")

conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
iou_threshold = st.slider("IOU threshold", 0.1, 1.0, 0.45, 0.05)

uploaded_files = st.file_uploader(
    "Choose image files", type=["png", "jpg", "jpeg", "bmp", "tiff"], accept_multiple_files=True
)

if uploaded_files:
    overall_counts = {name: 0 for name in model.names.values()}
    results_rows = []

    for uploaded in uploaded_files:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.warning(f"Skipping {uploaded.name} - could not read image")
            continue

        predictions = model.predict(image, conf=conf_threshold, iou=iou_threshold)
        result = predictions[0]
        per_image_counts = count_classes(result, model.names)

        for name, count in per_image_counts.items():
            overall_counts[name] = overall_counts.get(name, 0) + count

        annotated = result.plot()
        annotated_rgb = annotated[..., ::-1]

        st.subheader(uploaded.name)
        st.image(annotated_rgb, channels="RGB", use_container_width=True)

        results_rows.append({"image": uploaded.name, **per_image_counts})

    if results_rows:
        st.subheader("Per-image counts")
        st.dataframe(results_rows, use_container_width=True)

        st.subheader("Overall counts")
        st.dataframe([overall_counts], use_container_width=True)
else:
    st.info("Upload images to start detection.")
