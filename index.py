import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Set Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --- CombinedModel class definition ---
# --- Flask App Setup ---
app = Flask(__name__)
# Configure CORS to allow requests only from your frontend
CORS(app, resources={r"/*": {"origins": "https://skin-lesion-classifier.vercel.app"}})

class CombinedModel:
    def __init__(self, seg_model_path, cls_model_path, nlp_model_path, class_names):
        self.seg_model_path = seg_model_path
        self.cls_model_path = cls_model_path
        self.nlp_model_path = nlp_model_path
        self.class_names = class_names

        # Load models
        self.segmentation_model = tf.keras.models.load_model(seg_model_path)
        self.classification_model = tf.keras.models.load_model(cls_model_path)
        self.nlp_model = joblib.load(nlp_model_path)
    
    def preprocess_image(self, image_path):
        """
        Preprocess the image exactly as done during training:
        - Open with PIL
        - Convert to RGB
        - Resize to 224x224
        - Normalize to [0,1]
        """
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch, image
    
    def predict_cnn(self, image_path):
        """
        CNN prediction logic aligned with the working Streamlit code
        """
        img_batch, _ = self.preprocess_image(image_path)
        predictions = self.classification_model.predict(img_batch)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        all_probabilities = {class_name: float(prob) for class_name, prob in zip(self.class_names, predictions[0])}
        
        # Debugging
        print("CNN Raw Predictions:", predictions[0].tolist())
        print("CNN Predicted Class:", predicted_class, "Confidence:", confidence)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
    
    def predict_nlp(self, metadata):
        """
        NLP prediction based on metadata
        """
        input_df = pd.DataFrame([{k: metadata.get(k, '') for k in ['dx_type', 'age', 'sex', 'localization']}])
        nlp_pred = self.nlp_model.predict(input_df)[0]
        
        # Assuming NLP model outputs a class index or label; adjust based on your model
        try:
            nlp_pred_numeric = float(nlp_pred)
            predicted_class = self.class_names[int(nlp_pred_numeric)] if 0 <= int(nlp_pred_numeric) < len(self.class_names) else "Unknown"
            confidence = 1.0  # Default confidence if not provided by NLP model
        except (ValueError, TypeError):
            predicted_class = str(nlp_pred)
            confidence = None  # No confidence if output isn’t numeric
        
        print("NLP Prediction:", nlp_pred, "Predicted Class:", predicted_class)
        return {
            'predicted_class': predicted_class,
            'confidence': confidence
        }
    
    def predict_segmentation(self, image_path):
        """
        Segmentation prediction
        """
        img_batch, orig_img_pil = self.preprocess_image(image_path)
        mask_pred = self.segmentation_model.predict(img_batch)[0]
        mask_bin = (mask_pred > 0.5).astype(np.float32).squeeze()
        print("Segmentation Mask Mean:", np.mean(mask_bin))
        return mask_bin, orig_img_pil
    
    def predict(self, image_path, metadata):
        # CNN Prediction
        cnn_result = self.predict_cnn(image_path)
        
        # NLP Prediction
        nlp_result = self.predict_nlp(metadata)
        
        # Segmentation Prediction
        mask_bin, orig_img_pil = self.predict_segmentation(image_path)
        
        # Combined Prediction (choose higher confidence)
        combined_result = {}
        if nlp_result['confidence'] is not None and cnn_result['confidence'] < nlp_result['confidence']:
            combined_result = {
                'source': 'NLP',
                'predicted_class': nlp_result['predicted_class'],
                'confidence': nlp_result['confidence']
            }
        else:
            combined_result = {
                'source': 'CNN',
                'predicted_class': cnn_result['predicted_class'],
                'confidence': cnn_result['confidence']
            }
        
        return cnn_result, nlp_result, mask_bin, orig_img_pil, combined_result

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SEG_MODEL_PATH = "ap1_segmentation_model_final.keras"
CLS_MODEL_PATH = "C:/Users/HP/Downloads/classification_model_new_keras_ap1.keras"
NLP_MODEL_PATH = "C:/Users/HP/Downloads/path_to_trained_model.pkl"
CLASS_NAMES = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']

combined_model = CombinedModel(SEG_MODEL_PATH, CLS_MODEL_PATH, NLP_MODEL_PATH, CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image_file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        metadata = {k: request.form.get(k, "") for k in ["dx_type", "age", "sex", "localization"]}
        cnn_result, nlp_result, mask_bin, orig_img_pil, combined_result = combined_model.predict(image_path, metadata)
        
        # Visualization
        orig_img_rgb = np.array(orig_img_pil)
        overlay = orig_img_rgb.copy()
        overlay[mask_bin > 0.5] = [255, 0, 0]
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(orig_img_rgb)
        axs[0].axis("off")
        axs[0].set_title("Original Image")
        axs[1].imshow(mask_bin, cmap='gray')
        axs[1].axis("off")
        axs[1].set_title("Segmentation Mask")
        axs[2].imshow(overlay)
        axs[2].axis("off")
        axs[2].set_title(f"Prediction: {combined_result['predicted_class']} ({combined_result['source']})")
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        plot_image_base64 = base64.b64encode(buf.getvalue()).decode()
        
        # Prepare response
        response = {
            "cnn_result": {
                "predicted_class": cnn_result['predicted_class'],
                "confidence": cnn_result['confidence'],
                "all_probabilities": cnn_result['all_probabilities']
            },
            "nlp_result": {
                "predicted_class": nlp_result['predicted_class'],
                "confidence": nlp_result['confidence']
            },
            "segmentation_result": "Mask generated successfully",
            "combined_result": {
                "source": combined_result['source'],
                "predicted_class": combined_result['predicted_class'],
                "confidence": combined_result['confidence']
            },
            "plot_image": f"data:image/png;base64,{plot_image_base64}"
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
