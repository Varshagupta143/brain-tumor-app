  import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")

@st.cache_resource
def load_model():
    data = joblib.load("brain_tumor_model.pkl")
    return data["model"], data["scaler"], data["pca"], data["categories"]

model, scaler, pca, categories = load_model()

st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload a brain MRI scan and the AI will classify it.")
st.markdown("---")

st.markdown("### 📋 Classes this model detects:")
c1, c2, c3, c4 = st.columns(4)
c1.error("🔴 Glioma")
c2.warning("🟡 Meningioma")
c3.success("🟢 No Tumor")
c4.info("🔵 Pituitary")

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        img = np.array(image.convert("L"))
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        img = img.reshape(1, -1)

        img_scaled = scaler.transform(img)
        img_pca    = pca.transform(img_scaled)
        prediction = model.predict(img_pca)[0]
        class_name = categories[prediction]

    st.markdown("---")
    st.markdown("## 🎯 Result")

    if "notumor" in class_name.lower() or "no" in class_name.lower():
        st.success("✅ No Tumor Detected")
    else:
        st.error(f"⚠️ Tumor Detected: **{class_name.upper()}**")

    st.caption("⚠️ For educational purposes only. Consult a doctor for real diagnosis.")