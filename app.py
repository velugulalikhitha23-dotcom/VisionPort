import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch

#page configuration

st.set_page_config(
    page_title="Vision Port",
    page_icon=" ",
    layout="wide"
)

#title

st.title("Vision Port v0.1")

st.write(
    "GPU-Accelerated Traffic Object Detection"
)

device="cuda" if torch.cuda.is_available() else "cpu"

st.write(f"Running on:{device.upper()}")


#load YOLO model

@st.cache_resource
def load_model():
    model=YOLO("yolov8n.pt")

    return model

model=load_model()

#upload image

uploaded_file=st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png"]
)

#process image

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_container_width=True
    )

    #run configuration

    results=model.predict(
        source=image,
        device=device
    )

    # get detection result

    result_image=results[0].plot()

    # display detection result

    st.subheader("Detection Result")

    st.image(
        result_image,
        caption="Detected Objects",
        channels="BGR",
        use_container_width=True

    )