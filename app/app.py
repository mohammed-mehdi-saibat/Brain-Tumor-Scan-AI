import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# Class labels matching your training dataset order
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    """Load the trained Keras model."""
    model = tf.keras.models.load_model("brain_tumor_model.keras")
    return model

# App Header
st.title("🧠 Brain MRI Tumor Classification")
st.write("Upload a brain MRI scan image below to classify it using your trained deep learning model.")

# Load model with a spinner
with st.spinner("Loading trained model..."):
    model = load_model()

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a brain MRI scan image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Scan")
        st.image(image, caption="Input MRI Scan", use_container_width=True)
    
    # Preprocess image for the model
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Run prediction button
    if st.button("Classify Scan", type="primary"):
        with st.spinner("Analyzing scan..."):
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx]) * 100
            predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
        with col2:
            st.subheader("Prediction Result")
            
            # Display results cleanly based on class
            if predicted_class_name == "notumor":
                st.success(f"**Result:** No Tumor Detected\n\n**Confidence:** {confidence:.2f}%")
            else:
                st.warning(f"**Result:** {predicted_class_name.capitalize()} Tumor\n\n**Confidence:** {confidence:.2f}%")
            
            # Show breakdown across all classes
            st.write("### Confidence Breakdown:")
            for idx, class_name in enumerate(CLASS_NAMES):
                prob = float(predictions[0][idx]) * 100
                st.progress(prob / 100, text=f"{class_name.capitalize()}: {prob:.2f}%")