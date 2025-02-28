import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st

# Define the class indices with exact class names
class_indices = {
    "0": "Apple___Apple_scab",
    "1": "Apple___Black_rot",
    "2": "Apple___Cedar_apple_rust",
    "3": "Apple___healthy",
    "4": "Blueberry___healthy",
    "5": "Cherry_(including_sour)___Powdery_mildew",
    "6": "Cherry_(including_sour)___healthy",
    "7": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize)___Common_rust_",
    "9": "Corn_(maize)___Northern_Leaf_Blight",
    "10": "Corn_(maize)___healthy",
    "11": "Grape___Black_rot",
    "12": "Grape___Esca_(Black_Measles)",
    "13": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape___healthy",
    "15": "Orange___Haunglongbing_(Citrus_greening)",
    "16": "Peach___Bacterial_spot",
    "17": "Peach___healthy",
    "18": "Pepper,_bell___Bacterial_spot",
    "19": "Pepper,_bell___healthy",
    "20": "Potato___Early_blight",
    "21": "Potato___Late_blight",
    "22": "Potato___healthy",
    "23": "Raspberry___healthy",
    "24": "Soybean___healthy",
    "25": "Squash___Powdery_mildew",
    "26": "Strawberry___Leaf_scorch",
    "27": "Strawberry___healthy",
    "28": "Tomato___Bacterial_spot",
    "29": "Tomato___Early_blight",
    "30": "Tomato___Late_blight",
    "31": "Tomato___Leaf_Mold",
    "32": "Tomato___Septoria_leaf_spot",
    "33": "Tomato___Spider_mites Two-spotted_spider_mite",
    "34": "Tomato___Target_Spot",
    "35": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato___Tomato_mosaic_virus",
    "37": "Tomato___healthy"
}

# Remedies for each class
remedies = {
    "Apple___Apple_scab": {
        "Biological Control": "Apply a mixture of compost tea to enhance plant resistance.",
        "Organic Spray": "Use a copper-based fungicide to control the spread of the disease.",
        "Crop Management": "Remove fallen leaves and affected fruit regularly to reduce the spread."
    },
    "Apple___Black_rot": {
        "Biological Control": "Use beneficial fungi like Trichoderma to combat fungal infections.",
        "Organic Spray": "Apply neem oil or sulfur-based fungicide to infected areas.",
        "Crop Management": "Prune infected branches and dispose of them properly."
    },
    "Apple___Cedar_apple_rust": {
        "Biological Control": "Introduce rust-resistant apple varieties.",
        "Organic Spray": "Spray with a copper fungicide to prevent further spread.",
        "Crop Management": "Ensure good air circulation and remove fallen infected leaves."
    },
    "Apple___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Maintain healthy soil and provide adequate nutrients."
    },
    "Blueberry___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Keep soil acidic and ensure proper drainage."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "Biological Control": "Introduce predatory insects like ladybugs to control fungal pests.",
        "Organic Spray": "Spray with a mixture of baking soda and water.",
        "Crop Management": "Ensure proper spacing to reduce humidity and increase airflow."
    },
    "Cherry_(including_sour)___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure proper spacing and watering to promote healthy growth."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "Biological Control": "Use Trichoderma species to combat fungal infections.",
        "Organic Spray": "Apply a mixture of neem oil and water to control leaf spot.",
        "Crop Management": "Practice crop rotation and remove infected plant debris."
    },
    "Corn_(maize)___Common_rust_": {
        "Biological Control": "Use resistant maize varieties and introduce beneficial fungi.",
        "Organic Spray": "Spray with a sulfur-based fungicide to prevent rust.",
        "Crop Management": "Avoid planting maize in areas with poor air circulation."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Biological Control": "Introduce beneficial bacteria to control fungal spread.",
        "Organic Spray": "Apply copper fungicide to prevent further infection.",
        "Crop Management": "Ensure proper spacing and remove infected leaves."
    },
    "Corn_(maize)___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure adequate watering and nutrient levels."
    },
    "Grape___Black_rot": {
        "Biological Control": "Introduce beneficial fungi like Trichoderma to control rot.",
        "Organic Spray": "Use neem oil or sulfur-based fungicides to control the disease.",
        "Crop Management": "Prune infected areas and remove any fallen leaves."
    },
    "Grape___Esca_(Black_Measles)": {
        "Biological Control": "Use biofungicides to manage fungal infections.",
        "Organic Spray": "Spray with copper fungicide or sulfur-based sprays.",
        "Crop Management": "Ensure proper drainage and avoid excessive irrigation."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "Biological Control": "Use beneficial fungi like Trichoderma to combat fungal pathogens.",
        "Organic Spray": "Spray with a mixture of neem oil and water.",
        "Crop Management": "Prune affected areas and remove infected leaves."
    },
    "Grape___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure adequate irrigation and nutrient management."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "Biological Control": "Introduce natural predators like parasitic wasps to control insect vectors.",
        "Organic Spray": "Use organic pesticides like neem oil to control insect populations.",
        "Crop Management": "Prune infected branches and maintain proper plant spacing."
    },
    "Peach___Bacterial_spot": {
        "Biological Control": "Use beneficial bacteria like Bacillus to suppress pathogens.",
        "Organic Spray": "Apply copper-based fungicides or bactericides.",
        "Crop Management": "Prune infected areas and avoid overhead watering."
    },
    "Peach___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Maintain healthy soil and ensure adequate spacing."
    },
    "Pepper,_bell___Bacterial_spot": {
        "Biological Control": "Introduce beneficial microorganisms like Bacillus to suppress bacterial growth.",
        "Organic Spray": "Use neem oil or copper-based bactericides.",
        "Crop Management": "Avoid overhead irrigation and prune infected leaves."
    },
    "Pepper,_bell___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure proper spacing and soil fertility."
    },
    "Potato___Early_blight": {
        "Biological Control": "Introduce beneficial fungi to suppress early blight.",
        "Organic Spray": "Use a sulfur-based fungicide to manage the disease.",
        "Crop Management": "Practice crop rotation and remove infected plants."
    },
    "Potato___Late_blight": {
        "Biological Control": "Introduce beneficial microorganisms like Trichoderma to reduce late blight.",
        "Organic Spray": "Apply copper fungicides or sulfur-based sprays.",
        "Crop Management": "Ensure proper drainage and spacing to reduce humidity."
    },
    "Potato___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Maintain proper watering and fertilization practices."
    },
    "Raspberry___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure proper soil drainage and regular pruning."
    },
    "Soybean___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure good soil health and provide proper nutrients."
    },
    "Squash___Powdery_mildew": {
        "Biological Control": "Introduce natural predators like ladybugs to control fungal pests.",
        "Organic Spray": "Apply a mixture of baking soda and water to infected areas.",
        "Crop Management": "Ensure good air circulation and remove affected leaves."
    },
    "Strawberry___Leaf_scorch": {
        "Biological Control": "Introduce beneficial bacteria to control leaf scorch pathogens.",
        "Organic Spray": "Spray with a mixture of neem oil and water.",
        "Crop Management": "Ensure proper spacing and avoid overwatering."
    },
    "Strawberry___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure good soil drainage and proper plant care."
    },
    "Tomato___Bacterial_spot": {
        "Biological Control": "Use beneficial bacteria like Bacillus to suppress bacterial growth.",
        "Organic Spray": "Apply copper-based bactericides or neem oil.",
        "Crop Management": "Prune infected branches and ensure proper spacing."
    },
    "Tomato___Early_blight": {
        "Biological Control": "Introduce Trichoderma to reduce early blight spread.",
        "Organic Spray": "Use a sulfur-based fungicide to control the disease.",
        "Crop Management": "Practice crop rotation and remove infected leaves."
    },
    "Tomato___Late_blight": {
        "Biological Control": "Use Trichoderma to reduce the spread of late blight.",
        "Organic Spray": "Apply copper fungicide or sulfur-based fungicide.",
        "Crop Management": "Ensure good drainage and remove infected plant debris."
    },
    "Tomato___Leaf_Mold": {
        "Biological Control": "Introduce beneficial fungi to control mold growth.",
        "Organic Spray": "Spray with a mixture of neem oil and water.",
        "Crop Management": "Increase air circulation and reduce humidity around plants."
    },
    "Tomato___Septoria_leaf_spot": {
        "Biological Control": "Use Trichoderma to reduce fungal growth.",
        "Organic Spray": "Spray with copper fungicide or neem oil.",
        "Crop Management": "Prune affected leaves and ensure proper spacing."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "Biological Control": "Introduce natural predators like ladybugs to control spider mites.",
        "Organic Spray": "Spray with neem oil or insecticidal soap.",
        "Crop Management": "Ensure good air circulation and remove infected leaves."
    },
    "Tomato___Target_Spot": {
        "Biological Control": "Use Trichoderma to control fungal infections.",
        "Organic Spray": "Apply copper fungicide or neem oil.",
        "Crop Management": "Remove infected leaves and ensure good drainage."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "Biological Control": "Use natural predators to control insect vectors like whiteflies.",
        "Organic Spray": "Spray with insecticidal soap or neem oil.",
        "Crop Management": "Prune infected leaves and improve plant spacing."
    },
    "Tomato___Tomato_mosaic_virus": {
        "Biological Control": "Introduce natural predators to control insect vectors.",
        "Organic Spray": "Spray with insecticidal soap or neem oil.",
        "Crop Management": "Remove infected plants immediately to prevent further spread."
    },
    "Tomato___healthy": {
        "Biological Control": "No treatment needed for healthy plants.",
        "Organic Spray": "No treatment needed for healthy plants.",
        "Crop Management": "Ensure proper irrigation and nutrient management."
    }
}

    # Continue adding other diseases and remedies




# Load the trained model
model = tf.keras.models.load_model(r'C:\portfolio\Plant-Disease-Prediction-using-CNN\plant_leaf_disease_model.h5')  # Replace with your model's path

# Streamlit interface
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to predict the disease and view remedies.")

# Image upload functionality
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_image, target_size=(150, 150))  # Resize to the model's expected input size
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image for the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image if needed (depending on model training)
    
    # Make a prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)  # Index of the class with highest probability
    
    # Logic for mapping to one class above
    if predicted_class_index == 0:
        # Keep the same class for class 0
        output_class = class_indices["0"]
    else:
        # Map to one class above
        output_class_index = predicted_class_index - 1
        output_class = class_indices[str(output_class_index)]
    
    # Display the prediction result
    st.write(f"The model predicts the image is of class: **{output_class}**")
    
    # Display remedies
    if output_class in remedies:
        st.write("### Remedies:")
        st.write(remedies[output_class])
    else:
        st.write("No specific remedies available for this class.")


