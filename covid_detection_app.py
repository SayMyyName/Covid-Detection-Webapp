import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array

model = tf.keras.models.load_model("model_file_path")

def preprocess_image(image, model_name_chosed = "VGG16(Best)"):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    if model_name_chosed == 'VGG16(Best)':
        image = tf.keras.applications.vgg16.preprocess_input(image)
    elif model_name_chosed == 'InceptionResNetV2':
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    elif model_name_chosed == 'InceptionV3':
        image = tf.keras.applications.inception_v3.preprocess_input(image)
    elif model_name_chosed == 'ResNet50' or model_name_chosed == 'ResNet101' or model_name_chosed == 'ResNet101' or model_name_chosed == 'ResNet152':
        image = tf.keras.applications.resnet50.preprocess_input(image)
    else:
        image = image / 255.0
    
    return image

def load_model(model_name_chosed):
    if model_name_chosed == 'VGG16(Best)':
        return tf.keras.models.load_model("model_file_path")
    elif model_name_chosed == 'InceptionResNetV2':
        return tf.keras.models.load_model("model_file_path")
    elif model_name_chosed == 'InceptionV3':
        return tf.keras.models.load_model("model_file_path")
    elif model_name_chosed == 'ResNet50':
        return tf.keras.models.load_model("model_file_path")
    elif model_name_chosed == 'ResNet101':
        return tf.keras.models.load_model("model_file_path")
    elif model_name_chosed == 'ResNet101':
        return tf.keras.models.load_model("model_file_path")
    elif model_name_chosed == 'ResNet152':
        return tf.keras.models.load_model("model_file_path")
    else:
        return tf.keras.models.load_model("model_file_path")

st.title("COVID-19 Prediction from Chest X-Ray")

model_name = st.selectbox("Choose a model", ['VGG16(Best)', 'InceptionResNetV2', 'InceptionV3', 'ResNet50', 'ResNet101', 'ResNet152', 'No Transfer Learning'])

model = load_model(model_name)

uploaded_file = st.file_uploader("Choose a chest X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded chest X-ray.', use_column_width=True)

    if st.button('Predict'):
        with st.spinner("Analyzing the image..."):
            prepped_image = preprocess_image(image, model_name)
            prediction = model.predict(prepped_image)
            result = "Positive for COVID-19" if prediction[0][0] > 0.1 else "Negative for COVID-19"
            st.write("Prediction:", result)
