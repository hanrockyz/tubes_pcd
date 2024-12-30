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
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Sidebar with app information
with st.sidebar:
    st.title("🌿 Plant Disease Detection")
    st.write("This app uses a trained Random Forest model with GLCM features to classify apple leaf diseases.")
    st.write("### Instructions:")
    st.write("1. Upload an image of an apple leaf (JPEG or PNG).")
    st.write("2. Wait for the app to process the image.")
    st.write("3. View the disease prediction result.")

# Main app interface
st.title("Plant Disease Detection with GLCM Features")
st.markdown(
    """
    Upload an apple leaf image below to predict its disease type. 
    The model is trained to classify the following conditions:
    - Healthy
    - Cedar Apple Rust
    - Black Rot
    - Apple Scab
    """
)

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("📷 Image uploaded successfully!")

    # Display a spinner while processing the image
    with st.spinner("Processing the image..."):
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
        st.image(image, caption="Uploaded Image", use_container_width=True, channels="RGB")

        # Display extracted features (optional)
        with st.expander("🧪 Extracted GLCM Features (Optional)"):
            st.write(features)

        # Display prediction result
        if predicted_label == "Healthy":
            st.success(f"🌱 Predicted Disease: {predicted_label} - The apple is healthy!")
        elif predicted_label == "Cedar_apple_rust":
            st.error(f"🍂 Predicted Disease: {predicted_label} - The apple leaf has Cedar Apple Rust disease.")
        elif predicted_label == "Black_rot":
            st.warning(f"🛑 Predicted Disease: {predicted_label} - The apple leaf has Black Rot disease.")
        elif predicted_label == "Apple_scab":
            st.warning(f"🍎 Predicted Disease: {predicted_label} - The apple leaf has Apple Scab disease.")
else:
    st.info("📥 Please upload an image to get started.")
