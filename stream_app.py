%%writefile app.py

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

eye_model = load_model("eye_model.h5")
mouth_model = load_model("mouth_model.h5")

st.title("Driver Drowsiness Detection")

uploaded_file = st.file_uploader("Upload Image")

def preprocess(img):

    img = cv2.resize(img,(224,224))
    img = img/255
    img = np.expand_dims(img,axis=0)

    return img


if uploaded_file:

    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image)

    processed = preprocess(img)

    eye_pred = eye_model.predict(processed)
    mouth_pred = mouth_model.predict(processed)

    eye_class = np.argmax(eye_pred)
    mouth_class = np.argmax(mouth_pred)

    eye_label = ["Closed","Open"][eye_class]
    mouth_label = ["no_yawn","yawn"][mouth_class]

    if eye_label == "Open" and mouth_label == "no_yawn":
        fatigue = "Alert"
    elif eye_label == "Open" and mouth_label == "yawn":
        fatigue = "Mild Fatigue"
    elif eye_label == "Closed":
        fatigue = "Severe Fatigue"

    st.write("Eye State:",eye_label)
    st.write("Mouth State:",mouth_label)
    st.write("Fatigue Stage:",fatigue)
