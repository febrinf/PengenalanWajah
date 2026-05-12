import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="VisionFace Lite",
    layout="wide"
)

# ======================================
# LOAD MODEL
# ======================================
model_otsu = load_model("model_otsu.h5")
model_adap = load_model("model_adap.h5")

class_names = ["Dela", "Dymas", "Febri", "Sephia", "Yusril"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ======================================
# PREPROCESS
# ======================================
def preprocess_face(image, mode="otsu"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None, None

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))

    if mode == "otsu":
        processed = cv2.threshold(
            face, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
    else:
        processed = cv2.adaptiveThreshold(
            face,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    face_3ch = np.stack((processed,) * 3, axis=-1)
    face_3ch = face_3ch.astype("float32") / 255.0
    face_3ch = np.expand_dims(face_3ch, axis=0)

    return processed, face_3ch, (x, y, w, h)


# ======================================
# SIDEBAR
# ======================================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio("Navigasi", ["Home", "Analisis"])

# ======================================
# HOME
# ======================================
if menu == "Home":

    st.title("🎯 VisionFace Lite")

    st.markdown(
        """
        ### Sistem Klasifikasi Wajah
        
        Sistem ini menggunakan:
        - Haar Cascade (Deteksi Wajah)
        - Otsu Thresholding
        - Adaptive Thresholding
        - VGG16 (Deep Learning)

        ### Alur Sistem:
        1. Upload gambar
        2. Sistem mendeteksi wajah
        3. Preprocessing (Otsu & Adaptive)
        4. Klasifikasi wajah
        """
    )

# ======================================
# ANALISIS (UPLOAD + HASIL)
# ======================================
elif menu == "Analisis":

    st.title("🔍 Analisis Wajah")

    uploaded = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        file_bytes = np.asarray(
            bytearray(uploaded.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, caption="Gambar Input", width=300)

        if st.button("🚀 Proses Sekarang"):

            col1, col2 = st.columns(2)

            # ======================
            # OTSU
            # ======================
            face_o, tensor_o, box = preprocess_face(img, "otsu")

            with col1:
                st.subheader("Otsu")

                if face_o is not None:
                    pred_o = model_otsu.predict(tensor_o)
                    cls_o = class_names[np.argmax(pred_o)]
                    conf_o = np.max(pred_o)

                    st.image(face_o, width=250)
                    st.success(f"{cls_o}")
                    st.write(f"Confidence: {conf_o*100:.2f}%")
                else:
                    st.error("Wajah tidak terdeteksi")

            # ======================
            # ADAPTIVE
            # ======================
            face_a, tensor_a, _ = preprocess_face(img, "adaptive")

            with col2:
                st.subheader("Adaptive")

                if face_a is not None:
                    pred_a = model_adap.predict(tensor_a)
                    cls_a = class_names[np.argmax(pred_a)]
                    conf_a = np.max(pred_a)

                    st.image(face_a, width=250)
                    st.success(f"{cls_a}")
                    st.write(f"Confidence: {conf_a*100:.2f}%")
                else:
                    st.error("Wajah tidak terdeteksi")

            # ======================
            # BOUNDING BOX
            # ======================
            if box is not None:
                x, y, w, h = box
                img_box = img_rgb.copy()
                cv2.rectangle(img_box, (x, y), (x+w, y+h), (0,255,0), 2)

                st.subheader("📍 Deteksi Wajah")
                st.image(img_box, width=300)