import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Use Streamlit's cache to avoid reloading model every time
@st.cache_resource
def load():
    return load_model("flower_model.h5")

model = load()
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def predict(img):
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return classes[np.argmax(predictions)]

# Streamlit UI
st.title("🌸 Flower Classifier")
uploaded_file = st.file_uploader("Upload an image of a flower", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    label = predict(image)
    st.success(f"This looks like a **{label}**!")
