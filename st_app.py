import streamlit as st
import os
import io
import base64
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from PIL import Image

st.title("Skin Lesion Prediction App")

st.write("Upload an image and enter metadata for skin lesion classification.")

# Ensure the uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache the model loading so that they're loaded only once per session
@st.cache_resource
def load_models():
    segmentation_model = tf.keras.models.load_model("segmentation_model_final.keras")
    classification_model = tf.keras.models.load_model("not_overfitting_final.keras")
    nlp_model = joblib.load("path_to_trained_model.pkl")
    return segmentation_model, classification_model, nlp_model

segmentation_model, classification_model, nlp_model = load_models()

# Define class labels for the CNN classifier
CLASS_LABELS = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

# Inputs: File uploader and metadata
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
dx_type = st.text_input("Diagnosis Type (dx_type)")
age = st.text_input("Age")
sex = st.selectbox("Sex", options=["", "Male", "Female"])
localization = st.text_input("Localization")

if st.button("Predict"):
    if uploaded_file is None:
        st.error("Please upload an image.")
    else:
        try:
            # Convert uploaded file to OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            orig_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if orig_img is None:
                st.error("Could not process the image. Please try a different file.")
            else:
                # Optionally, save the image locally (if needed for debugging)
                image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Preprocess the image
                orig_img = cv2.resize(orig_img, (224, 224))
                orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                img_norm = orig_img.astype(np.float32) / 255.0
                img_batch = np.expand_dims(img_norm, axis=0)

                # 1) Segmentation using U-Net model
                mask_pred = segmentation_model.predict(img_batch)[0]
                mask_bin = (mask_pred > 0.5).astype(np.float32).squeeze()

                # 2) Create masked image for CNN classification
                mask_3ch = np.repeat(mask_bin[:, :, np.newaxis], 3, axis=-1)
                masked_img = img_norm * mask_3ch

                # 3) CNN Classification
                cnn_probs = classification_model.predict(np.expand_dims(masked_img, axis=0))[0]
                top_idx = int(np.argmax(cnn_probs))
                top_class_label = CLASS_LABELS[top_idx] if top_idx < len(CLASS_LABELS) else "Unknown"
                top_class_conf = float(cnn_probs[top_idx])
                all_class_probs_str = ", ".join(
                    f"{CLASS_LABELS[i]}: {cnn_probs[i]:.4f}" for i in range(len(CLASS_LABELS))
                )

                # 4) NLP Prediction using metadata
                input_df = pd.DataFrame([{
                    "dx_type": dx_type,
                    "age": age,
                    "sex": sex,
                    "localization": localization
                }])
                nlp_pred = nlp_model.predict(input_df)[0]

                # 5) Combine predictions if possible
                try:
                    nlp_pred_numeric = float(nlp_pred)
                    combined_pred = (cnn_probs + nlp_pred_numeric) / 2.0
                    final_idx = int(np.argmax(combined_pred))
                    final_label = CLASS_LABELS[final_idx]
                except (ValueError, TypeError):
                    final_label = str(nlp_pred)

                # 6) Create a visualization (2x2 subplots)
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                # Masked image (CNN input)
                axs[0, 0].imshow(masked_img)
                axs[0, 0].axis("off")
                axs[0, 0].set_title("CNN", fontsize=12, pad=10)
                # Original image
                axs[0, 1].imshow(orig_img_rgb)
                axs[0, 1].axis("off")
                axs[0, 1].set_title("Original", fontsize=12, pad=10)
                # Segmentation overlay on original image
                overlay = orig_img_rgb.copy()
                overlay[mask_bin > 0.5] = [255, 0, 0]
                axs[1, 0].imshow(overlay)
                axs[1, 0].axis("off")
                axs[1, 0].set_title("Segmentation", fontsize=12, pad=10)
                # Combined view (for example, original)
                axs[1, 1].imshow(orig_img_rgb)
                axs[1, 1].axis("off")
                axs[1, 1].set_title("Combined", fontsize=12, pad=10)
                plt.tight_layout(pad=2.0)

                # Encode the figure as a PNG image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                plt.close(fig)
                plot_image = Image.open(buf)

                # Display the results
                st.image(plot_image, caption="Visualization", use_column_width=True)
                st.subheader("Predictions:")
                st.write(f"**Predicted Class:** {top_class_label} (Confidence: {top_class_conf:.2f})")
                st.write(f"**All Class Probabilities:** {all_class_probs_str}")
                st.write(f"**NLP Prediction:** {nlp_pred}")
                st.write("**Segmentation:** U-Net Segmentation Applied")
                st.write(f"**Final Prediction:** {final_label}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
