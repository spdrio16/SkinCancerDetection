import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Pillow fix
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")
st.title("Skin Lesion Classifier — HAM10000")

MODEL_PATH = "model.h5"

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanocytic nevi",
    "Melanoma",
    "Vascular lesions"
]

# ----------------------------------------------------------
# AUTO-DETECT MODEL INPUT SIZE (FIXED VERSION)
# ----------------------------------------------------------
def auto_detect_size(model):
    shape = model.input_shape   # always safe

    # Case 1: CNN → (None, H, W, C)
    if len(shape) == 4:
        _, h, w, c = shape
        if h is not None and w is not None:
            return (w, h)  # width, height

    # Case 2: Dense → (None, N)
    if len(shape) == 2:
        flat = shape[1]             # N = H*W*C
        pixels = flat // 3          # H*W

        for h in range(40, 400):
            if pixels % h == 0:
                w = pixels // h
                return (w, h)

    return (224, 224)


@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return load_model(path)


# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
model = None

if st.button("Load model"):
    try:
        model = load_keras_model(MODEL_PATH)
        st.success("Model loaded")

        try:
            buf = io.StringIO()
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            st.text_area("Model Summary", buf.getvalue(), height=240)
        except:
            pass

    except Exception as e:
        st.error(f"{e}")

# Auto-load
if model is None and os.path.exists(MODEL_PATH):
    try:
        model = load_keras_model(MODEL_PATH)
        st.info("Auto-loaded model")
    except:
        pass

# Detect input size after load
if model:
    DETECTED_WIDTH, DETECTED_HEIGHT = auto_detect_size(model)
    st.info(f"Model expects image size: {DETECTED_WIDTH} × {DETECTED_HEIGHT}")
else:
    DETECTED_WIDTH, DETECTED_HEIGHT = (224, 224)


# ----------------------------------------------------------
# PREPROCESS IMAGE
# ----------------------------------------------------------
def preprocess_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = target_size
    img = ImageOps.fit(img, (w, h), method=RESAMPLE_LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ----------------------------------------------------------
# UI LAYOUT
# ----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_column_width=True)
    else:
        st.info("Upload an image to begin.")

with col2:
    st.markdown("### Example Lesion Classes")
    if os.path.exists("category_samples.png"):
        st.image("category_samples.png", use_column_width=True)
    else:
        st.write("No example image found.")


# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------
def predict(img):
    if model is None:
        raise RuntimeError("Model not loaded")

    x = preprocess_image(img, (DETECTED_WIDTH, DETECTED_HEIGHT))
    preds = model.predict(x)

    if preds.ndim == 2 and preds.shape[1] == len(CLASS_NAMES):
        probs = tf.nn.softmax(preds[0]).numpy()
    else:
        probs = preds.flatten()
        probs = probs / probs.sum()

    return probs


if uploaded:
    if st.button("Predict"):
        try:
            if model is None:
                model = load_keras_model(MODEL_PATH)

            probs = predict(image)
            idxs = np.argsort(probs)[::-1][:3]

            st.success("Top Predictions")
            for i in idxs:
                st.write(f"{CLASS_NAMES[i]} — {probs[i]*100:.2f}%")

            import pandas as pd
            df = pd.DataFrame({
                "class": CLASS_NAMES,
                "probability": (probs * 100).round(2)
            }).sort_values("probability", ascending=False)

            st.dataframe(df, height=260)

            # ----------------------------------------------------------
            # ✅ CONCLUSION (ADDED HERE)
            # ----------------------------------------------------------
            top_index = np.argmax(probs)
            top_class = CLASS_NAMES[top_index]
            top_conf = probs[top_index] * 100

            st.subheader("Conclusion")
            st.write(
                f"The model identifies **{top_class}** as the most probable condition "
                f"with a confidence of **{top_conf:.2f}%**. "
                "This indicates that the uploaded skin lesion most closely matches this disease category."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
