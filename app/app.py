import streamlit as st
import os
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# Load the trained Random Forest model
model = joblib.load("random_forest_glcm_model.pkl")

# Label mapping (same as the training part)
labels = {
    0: "Healthy",
    1: "Cedar_apple_rust",
    2: "Black_rot",
    3: "Apple_scab"
}

# GLCM feature extraction function
def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))  # Resize for consistency
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(np.mean(graycoprops(glcm, prop)))
    return features

# Streamlit interface
st.title("Apple Disease Prediction")

st.write("Upload an apple leaf image to predict its disease based on GLCM features.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Check if the image is in RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Extract GLCM features
    features = extract_glcm_features(image)

    # Reshape features for prediction
    features = np.array(features).reshape(1, -1)

    # Predict disease
    prediction = model.predict(features)
    predicted_label = labels[prediction[0]]

    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Display prediction
    st.write(f"Predicted Disease: {predicted_label}")

    # Optionally display additional details based on the prediction
    if predicted_label == "Healthy":
        st.write("The apple is healthy!")
    elif predicted_label == "Cedar_apple_rust":
        st.write("The apple leaf has Cedar Apple Rust disease.")
    elif predicted_label == "Black_rot":
        st.write("The apple leaf has Black Rot disease.")
    elif predicted_label == "Apple_scab":
        st.write("The apple leaf has Apple Scab disease.")
