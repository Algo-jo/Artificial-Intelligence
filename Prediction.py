import numpy as np
from PIL import Image
from SavingModel import loadModel
import json
import os

# SIMPLE SINGLE IMAGE PREDICTION
#===============================================================================
def predict_image(image_path, model_path='stunting_model.keras'):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return None
    
    # Load model (stunting_model.keras)
    print("\n[1/2] Loading model and class indinces...")
    model = loadModel(model_path)

    # Load class names (from class_indices.json)
    #=============================================================================
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        classes = {v: k for k, v in class_indices.items()}
    except:
        classes = {0: 'normal', 1: 'stunting'}
    
    # Preprocess image
    #==============================================================================
    print(f"Processing image: {os.path.basename(image_path)}")
    img = Image.open(image_path)
    img = img.convert('L')  # Grayscale. kalo warna gajelas
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=-1)  
    img_array = np.expand_dims(img_array, axis=0)  
    
    # Prediction of stunting or not
    #==============================================================================
    print("\n[2/2] Predicting...")
    probability = model.predict(img_array, verbose=0)[0][0]
    predicted_class = 1 if probability > 0.5 else 0
    confidence = probability if predicted_class == 1 else (1 - probability)
    
    # Display result
    #==============================================================================
    print("\n" + "="*60)
    print("STUNTING DETECTION RESULT")
    print("="*60)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {classes[predicted_class].upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Score: {probability:.4f}")
    
    if predicted_class == 1:
        print("\nWARNING: Stunting detected!")
    else:
        print("\nNormal growth detected")
    
    print("="*60 + "\n")
    
    return {
        'image': os.path.basename(image_path),
        'prediction': classes[predicted_class],
        'confidence': confidence,
        'is_stunting': predicted_class == 1,
        'raw_score': probability
    }

# RUN PREDICTION
#===============================================================================
if __name__ == "__main__": # Program ini input path image manual. Nanti harus dikembangin
    image_path = input("Enter the path to the image file: ") 
    result = predict_image(image_path)