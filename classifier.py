import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))


# Function to load the selected model
def load_model(model_name):
    model_path = f"{working_dir}/models/{model_name}"
    # Load the selected pre-trained model
    model = tf.keras.models.load_model(model_path)
    return model


# Define class labels for Galaxy Morphological dataset
class_names = [
    "Disturbed",
    "Merging",
    "Round_Smooth",
    "In-between_Round_Smooth",
    "Cigar_Shaped_Smooth",
    "Barred_Spiral",
    "Unbarred_Tight_Spiral",
    "Unbarred_Loose_Spiral",
    "Edge-on_without_Bulge",
    "Edge-on_with_Bulge",
]

# Define the redshift model name
redshift_model_name = "galaxy_redshift_model_v1.h5"


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((64, 64))  # Resize to match the input size of your model
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape((1, 64, 64, 3))  # Add batch dimension
    return img_array


# Galaxy Classifier and Redshift Predictor function for Streamlit app
def galaxy_classifier():
    st.markdown(
        """
        <h1 style="text-align: center;">Galaxy Morphology Classifier and Redshift Predictor</h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align: center; font-weight: bold; font-size: 18px;">
            Dynamically predict galaxy types, redshift values, or star formation rates using machine learning. The model adjusts predictions based on updated astronomical observations, improving accuracy over time <br>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Select model for morphology classification
    model_option = st.selectbox(
        "Select Morphology Classification Model",
        ["ResNet50_model_v1.h5", "VGG19_model_v1.h5", "EfficientNetB0_model_v2.h5"],
    )

    # Load selected morphology classification model
    model = load_model(model_option)

    # Load redshift model
    redshift_model = load_model(redshift_model_name)

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2, col3 = st.columns(3)

        with col1:
            resized_img = image.resize((256, 256))
            st.image(resized_img, caption="Uploaded Image")

        with col2:
            if st.button("Classify Galaxy and Predict Redshift"):
                # Preprocess the uploaded image
                img_array = preprocess_image(uploaded_image)

                # Make a prediction using the selected morphology classification model
                result = model.predict(img_array)
                predicted_class = np.argmax(result)
                prediction = class_names[predicted_class]

                # Calculate confidence score
                confidence_score = np.max(result) * 100

                # Display morphology prediction and confidence score
                st.success(f"Classification Prediction: {prediction}")
                st.info(f"Confidence: {confidence_score:.2f}%")

                # Predict redshift
                redshift_prediction = redshift_model.predict(img_array)
                st.success(f"Redshift Prediction: {redshift_prediction[0][0]:.4f}")

                with col3:

                    # Visualize prediction probabilities
                    fig, ax = plt.subplots()
                    ax.barh(class_names, result.flatten())
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("Class")
                    ax.set_title("Prediction Probabilities")
                    st.pyplot(fig)
