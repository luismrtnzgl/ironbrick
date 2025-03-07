import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils.model_utils import load_model
from app.predict import predict

MODEL_PATH = "app/model/modelo_lego_final.pth"
model = load_model(MODEL_PATH)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        predicted_class, confidence = predict(img_pil, model)

        cv2.putText(img, f"Set: {predicted_class} - {confidence:.2f}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

st.title("📹 Identificación de Sets en Tiempo Real")
webrtc_streamer(key="lego-camera", video_transformer_factory=VideoTransformer)
