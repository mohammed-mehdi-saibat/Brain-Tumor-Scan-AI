import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# DEFINE CLASS LABELS MATCHING TRAINING MAPPING ORDER
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_brain_tumor(img_path, model_path="brain_tumor_model.keras"):
    # VALIDATE PATHS
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Run python train.py first")
        return
    
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} not found.")
        return 
    
    # LOAD TRAINED MODEL
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # LOAD AND PREPROCESS THE TARGET IMAGE
    print(f"Preprocessing image {img_path}...")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    # ADD BATCH DIMENSION -> SHAPE: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # NORMALIZE PIXEL VALUES 
    img_array = img_array / 255.0
    
    # RUN INFERENCE
    print("Running inference...")    
    predictions = model.predict(img_array)
    
    predicted_index = np.argmax(predictions[0])
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    
    # DISPLAY FORMATTED RESULTS
    print("\n" + "="*45)
    print("            BRAIN TUMOR PREDICTED RESULT ")
    print("="*45)
    print(f"Prediction: {predicted_label.upper()} ({confidence:.2f}% confidence)")
    print("-"* 45)
    print("Full Probability Breakdown:")
    for idx, score in enumerate(predictions[0]):
        print(f"    .{CLASS_LABELS[idx]:<12}: {score * 100:.2f}%")
    print("="*45 + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a single brain MRI image using the trained CNN model.")
    parser.add_argument("image_path", type=str, help="Path to the brain MRI scan image file.")
    parser.add_argument("--model", type=str, default="brain_tumor_model.keras", help="Path to the saved model file.")
    
    args = parser.parse_args()
    predict_brain_tumor(args.image_path, args.model)