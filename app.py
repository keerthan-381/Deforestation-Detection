import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import normalize

# Load the trained model
import numpy as np
import tensorflow as tf

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(tf.clip_by_value(y_true + y_pred, 0, 1))
        iou = (intersection + 1e-15) / (union + 1e-15)
        return iou
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


model = load_model('ResUNET-1.hdf5', custom_objects={"iou": iou})

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))
    image = normalize(image, axis=1)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return image

# Function to apply the model and generate predictions
def generate_predictions(image):
    predictions = model.predict(image)
    mask = (predictions[0, :, :, 0] > 0.2).astype(np.uint8) * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR
    return mask

# Streamlit app
def main():
    st.title("Mask Prediction")
    st.write("Upload an image and see the predicted mask")

    # Upload the original image
    uploaded_file = st.file_uploader("Choose the original image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the original image
        original_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Preprocess the original image
        preprocessed_image = preprocess_image(original_image)

        # Generate predictions
        mask = generate_predictions(preprocessed_image)

        # Display the original image and predicted mask
        st.subheader("Original Image")
        st.image(original_image, channels="RGB")

        st.subheader("Predicted Mask")
        st.image(mask, channels="RGB")

        # Extract the mask image based on the name
        image_name = uploaded_file.name
        mask_path = os.path.join(r"C:\Users\Dell\Downloads\data_shaffled_8_04_2021\masks", image_name)  # Replace "mask_folder" with the actual path to your mask images folder

        if os.path.exists(mask_path):
            # Read the actual mask image
            actual_mask = cv2.imread(mask_path)

            # Display the actual mask image
            st.subheader("Actual Mask")
            st.image(actual_mask, channels="RGB")
        else:
            st.warning("Actual mask image not found.")

if __name__ == "__main__":
    main()
