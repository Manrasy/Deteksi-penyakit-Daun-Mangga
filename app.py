import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

IMG_SIZE_FEAT = (300, 300)

st.set_page_config(page_title="Deteksi Penyakit Daun Mangga", page_icon="🥭")


@st.cache_resource
def load_artifacts():
    clf = joblib.load("model/mlp_model.joblib")
    scaler = joblib.load("model/scaler.joblib")
    le = joblib.load("model/label_encoder.joblib")
    cols = joblib.load("model/feature_cols.joblib")
    return clf, scaler, le, cols


def extract_features(img_bgr):
    """Harus identik dengan extract_handcrafted_features di notebook."""
    img = cv2.resize(img_bgr, IMG_SIZE_FEAT)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = perimeter = aspect_ratio = eccentricity = 0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        m = cv2.moments(c)
        den = (m['mu20'] + m['mu02']) ** 2
        if den != 0:
            val = ((m['mu20'] - m['mu02']) ** 2 - 4 * m['mu11'] ** 2) / den
            eccentricity = np.sqrt(abs(val))

    return {
        'R_Mean': rgb[:, :, 0].mean(), 'G_Mean': rgb[:, :, 1].mean(), 'B_Mean': rgb[:, :, 2].mean(),
        'H_Mean': hsv[:, :, 0].mean(), 'S_Mean': hsv[:, :, 1].mean(), 'V_Mean': hsv[:, :, 2].mean(),
        'Contrast': graycoprops(glcm, 'contrast')[0, 0],
        'Dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'Homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'Energy': graycoprops(glcm, 'energy')[0, 0],
        'Correlation': graycoprops(glcm, 'correlation')[0, 0],
        'Area': area, 'Perimeter': perimeter,
        'Aspect_Ratio': aspect_ratio, 'Eccentricity': eccentricity,
    }


st.title("🥭 Deteksi Penyakit Daun Mangga")
st.write("Unggah foto daun mangga, model akan memprediksi penyakitnya.")

clf, scaler, le, cols = load_artifacts()

file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
if file:
    pil = Image.open(file).convert("RGB")
    st.image(pil, caption="Gambar yang diunggah", use_container_width=True)

    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    feats = pd.DataFrame([extract_features(img_bgr)])[cols]
    proba = clf.predict_proba(scaler.transform(feats))[0]

    top = int(np.argmax(proba))
    st.success(f"Prediksi: **{le.classes_[top]}** ({proba[top]*100:.1f}%)")

    st.subheader("Probabilitas semua kelas")
    st.bar_chart(pd.Series(proba, index=le.classes_))
