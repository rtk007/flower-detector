import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("flower_model.h5")
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def predict(img):
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return classes[np.argmax(pred)]

st.title("🌸 Flower Classifier")
file = st.file_uploader("Upload a flower image", type=["jpg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    result = predict(image)
    st.success(f"This looks like a **{result}**!")
