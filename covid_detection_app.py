import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model("model_file_path")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def predict(image):
    prepped_image = tf.keras.applications.vgg16.preprocess_input(image)
    prediction = model.predict(prepped_image)
    return prediction

st.title("COVID-19 Prediction from Chest X-Ray")

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded chest X-ray.', use_column_width=True)

    if st.button('Predict'):
        with st.spinner("Analyzing the image..."):
            prediction = predict(image)
            result = "Positive for COVID-19" if prediction[0][0] > 0.1 else "Negative for COVID-19"
            st.write("Prediction:", result)
