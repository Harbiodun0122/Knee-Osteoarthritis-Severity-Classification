import os
import torch
import gdown
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from detectron2 import model_zoo
from detectron2.config import get_cfg
from tensorflow.keras.models import Model
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from tensorflow.keras.preprocessing import image
from detectron2.utils.visualizer import ColorMode, Visualizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img


st.set_page_config(page_title="Medical Image Classification", page_icon="🩺")

@st.cache_data
def download_model(file_id: str, filename: str):
    """Download a file from Google Drive if it doesn't already exist locally."""
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename


# Define model info
MODEL_1_ID = "1-sUyuPxgA-Oz8insa7Z617pQ-jL3USNT"  
MODEL_2_ID = "1poRrN4TsVmU1zD8BxX2BaIIHoJrgbTp7"  

MODEL_1_PATH = download_model(MODEL_1_ID, "improved-48-0.84.keras")
MODEL_2_PATH = download_model(MODEL_2_ID, "model_final.pth")    


# Load your trained model
knee_model = tf.keras.models.load_model(MODEL_1_PATH)


# Load the Configuration and Model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
cfg.MODEL.WEIGHTS = MODEL_2_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9   # Set a threshold for the detection confidence as 90%
cfg.MODEL.DEVICE = "cpu"  
predictor = DefaultPredictor(cfg)


def main():
    # Sidebar
    st.sidebar.image("11.jpeg", use_column_width=True)
    st.sidebar.markdown("""
    ### Knee Osteoarthritis Severity Classifier
    Upload an image and let the model detect and classify the condition of the knee joint.

    _Built with Detectron2 & Xception._
    """)

    # Top bar
    st.markdown("""
        <style>
        .top-bar {
            background-color: #004d7a;
            padding: 20px 10px;
            color: white;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .top-bar h1 {
            color: white;
            margin: 0;
            font-size: 2.4em;
        }
        .top-bar p {
            margin: 5px 0 0;
            font-size: 1.1em;
        }
        </style>
        <div class="top-bar">
            <h1>Knee Osteoarthritis Detection</h1>
            <p>Automatically detect and classify the severity of osteoarthritis from knee images.</p>
        </div>
    """, unsafe_allow_html=True)

    # Single Page App
    knee_osteoarthritis_classification()

    # Footer
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #004d7a;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 0.9em;
        }
        </style>
        <div class="footer">
            © 2025 Ikuobolati Abiodun Project
        </div>
    """, unsafe_allow_html=True)

def knee_osteoarthritis_classification():
    st.header("Upload Knee X-ray Image")
    knee_image = st.file_uploader("Upload a knee X-ray image", type=["jpg", "png", "jpeg"], key="knee")

    col1, col2 = st.columns(2)
    if knee_image is not None:  
      with col1:
          # Load and show uploaded image
          uploaded_image = Image.open(knee_image).convert("RGB")
          st.image(uploaded_image.resize((300, 300)), caption="Uploaded Image", use_container_width=False)

          im_array = np.array(uploaded_image)

      with col2:
        try:
          with st.spinner("🔍 Detecting knee joint..."):
            outputs = predictor(im_array)
            boxes = outputs["instances"].pred_boxes
            box = boxes[0].tensor.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)
            cropped_array = im_array[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped_array)

          # Show cropped image immediately
          st.image(cropped_image.resize((300, 300)), caption="Detected Knee Joint", use_container_width=False)
          
          if len(boxes) > 1:
            st.info("Extracted only the right knee joint from the detected knee joints.")
          elif len(boxes) == 0:
            st.error("No knee joint detected.")

        except Exception as e:
            st.error(f"Error detecting knee joint: {e}")
            return


      # Classification
      try:
          with st.spinner("Classifying..."):
            st.spinner('Classifying...')
            resized_input = tf.image.resize(cropped_array, (160, 335))
            model_input = np.expand_dims(resized_input / 255.0, 0)
            prediction = knee_model.predict(model_input)

            class_map = {0: "Normal", 1: "Doubtful", 2: "Mild", 3: "Moderate", 4: "Severe"}
            predicted_class = np.argmax(prediction)
            confidence = round(prediction[0][predicted_class] * 100, 2)

          st.success(f"**Prediction:** {class_map[predicted_class]}")
          st.info(f"**Confidence:** {confidence:.2f}%")

      except Exception as e:
          st.error(f"Error during classification: {e}")


if __name__ == "__main__":
    main()
