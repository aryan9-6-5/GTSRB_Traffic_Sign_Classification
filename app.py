import os
import cv2
import json
import time
import numpy as np
import requests
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()
try:
    st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="wide")
except Exception:
    pass

MODEL_PATH = "model.h5"
INPUT_SIZE = (32, 32)
TOP_K = 5

LABELS = {
    0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def get_model():
    if "model" not in st.session_state:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file 'model.h5' not found next to app.py.")
            st.stop()
        st.session_state.model = load_model(MODEL_PATH)
    return st.session_state.model

def to_bgr(np_img: np.ndarray) -> np.ndarray:
    if np_img.ndim == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    if np_img.shape[2] == 4:
        return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def preprocessing(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return eq.astype(np.float32) / 255.0

def prepare_tensor(img_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(img_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    pre = preprocessing(img_resized)
    return np.expand_dims(pre, axis=(0, -1))

def topk_from_probs(probs: np.ndarray, k: int = 5):
    k = int(min(k, probs.size))
    idxs = np.argpartition(-probs, k - 1)[:k]
    idxs = idxs[np.argsort(-probs[idxs])]
    return [(int(i), float(probs[i])) for i in idxs]

def predict(img_bgr: np.ndarray):
    model = get_model()
    x = prepare_tensor(img_bgr)
    preds = model.predict(x, verbose=0)
    probs = preds[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    return class_idx, confidence, probs

def gemini_generate(contents):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "Set GEMINI_API_KEY environment variable."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents}
    try:
        r = requests.post(url, params={"key": api_key}, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
    except Exception as e:
        return f"Error: {e}"

st.markdown("""
<style>
.block-container{padding-top:2rem;padding-bottom:2rem}
.kpi{border-radius:16px;padding:14px 16px;border:1px solid rgba(0,0,0,.08);box-shadow:0 1px 3px rgba(0,0,0,.06)}
.pred{font-weight:600;font-size:1.05rem}
.conf{opacity:.9}
.chatbox{border:1px solid rgba(0,0,0,.08);border-radius:14px;padding:12px;height:420px;overflow:auto;background:rgba(0,0,0,.02)}
.msgu{margin:6px 0;font-weight:600}
.msga{margin:6px 0}
.btnrow{display:flex;gap:.5rem;align-items:center}
hr{margin:.75rem 0}
</style>
""", unsafe_allow_html=True)

st.title("🚦 Traffic Sign Recognition")
st.caption("Upload → preprocess → predict. Includes a Gemini-powered chatbot for help and explanations.")

left, right = st.columns([2,1], gap="large")

with left:
    tabs = st.tabs(["Predict", "Preprocess", "Top-K"])
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded is not None:
        t0 = time.time()
        try:
            pil = Image.open(uploaded).convert("RGBA") if uploaded.type.endswith("png") else Image.open(uploaded)
            np_img = np.array(pil)
        except Exception as e:
            st.error(f"Failed to read image: {e}")
            st.stop()
        st.image(np_img, caption="Uploaded image", width=480)
        img_bgr = to_bgr(np_img)
        try:
            class_idx, conf, probs = predict(img_bgr)
            label = LABELS.get(class_idx, f"Class {class_idx}")
            dt = max(time.time() - t0, 0.001)
            kpi1, kpi2 = st.columns(2)
            with kpi1:
                st.markdown(f"<div class='kpi'><div class='pred'>Prediction</div><div>{label}</div></div>", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"<div class='kpi'><div class='pred'>Confidence</div><div class='conf'>{conf*100:.2f}%</div></div>", unsafe_allow_html=True)
            st.progress(min(1.0, conf))
            with tabs[1]:
                pre32 = cv2.resize(img_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
                pre32 = preprocessing(pre32)
                display_pre = (pre32 * 255).clip(0, 255).astype(np.uint8)
                st.image(display_pre, caption="Preprocessed 32×32", width=256)
            with tabs[2]:
                for i, (idx, p) in enumerate(topk_from_probs(probs, k=TOP_K), start=1):
                    st.write(f"{i}. {LABELS.get(idx, idx)} — {p*100:.2f}%")
            st.caption(f"Latency: {dt:.3f}s")
            st.session_state["last_pred"] = {"label": label, "confidence": conf, "topk": [(LABELS.get(i,i),p) for i,p in topk_from_probs(probs, k=TOP_K)]}
        except Exception as e:
            st.exception(e)
    else:
        st.info("Upload a JPG/PNG image to start.")

with right:
    st.subheader("💬 Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = [
            {"role": "assistant", "text": "Hello! Ask me about predictions, sign meanings, or troubleshooting."}
        ]

    user_input = st.text_input("Type your question")
    colA, colB = st.columns([1,1])
    send = colA.button("Send")
    clear = colB.button("Clear")

    st.markdown("<div class='chatbox'>", unsafe_allow_html=True)
    for m in st.session_state.chat:
        if m["role"] == "user":
            st.markdown(f"<div class='msgu'>You: {m['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msga'>Bot: {m['text']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.chat = [{"role": "assistant", "text": "Cleared. How can I help?"}]
        st.experimental_rerun()

    if send and user_input.strip():
        st.session_state.chat.append({"role": "user", "text": user_input.strip()})
        primer = {"role": "user", "parts": [{"text": "You are a concise assistant for a traffic sign recognition app. If a last prediction is provided, use it."}]}
        conv = []
        for m in st.session_state.chat:
            if m["role"]=="user":
                conv.append({"role":"user","parts":[{"text":m["text"]}]})
            else:
                conv.append({"role":"model","parts":[{"text":m["text"]}]})
        if "last_pred" in st.session_state:
            lp = st.session_state["last_pred"]
            lp_text = f"Last prediction: {lp['label']} with confidence {lp['confidence']*100:.2f}%. Top-K: " + ", ".join([f"{n} {p*100:.1f}%" for n,p in lp["topk"]])
            conv.append({"role":"user","parts":[{"text":lp_text}]})
        with st.spinner("Thinking..."):
            reply = gemini_generate([primer] + conv[-12:])
        st.session_state.chat.append({"role": "assistant", "text": reply})
        st.experimental_rerun()
