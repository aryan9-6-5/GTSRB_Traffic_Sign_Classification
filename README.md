# 🚦 Traffic Sign Recognition + Gemini Chatbot

A **traffic sign recognition app** built with **Streamlit** and **TensorFlow/Keras**, enhanced with a **Gemini-powered chatbot** that explains predictions, traffic sign meanings, and troubleshooting tips.  
Optimized to work with **Streamlit 1.20.0** (no `st.chat_*` APIs).

---

## ✨ Features

- Upload a JPG/PNG traffic sign image and get:
  - **Predicted sign label**
  - **Confidence score + progress bar**
  - **Top-K predictions**
  - **Preprocessed 32×32 input visualization**
- Built-in **Gemini chatbot** for Q&A:
  - Ask about prediction results  
  - Ask about sign meanings  
  - Get troubleshooting help  

---

## 📦 Installation

Clone your project and install dependencies.  
Your environment already has Streamlit + ML libs — you only need one extra package for the chatbot:

```bash
pip install -r requirements.txt
```

### 🔑 API Key Setup

Set your Gemini API key from Google AI Studio:

**.env file:**
```cmd
GEMINI_API_KEY=YOUR_KEY_HERE
```

---

## ▶️ Run the App

From the project folder containing `app.py` and `model.h5`:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🖼️ Usage

1. **Upload an image** (JPG/PNG) in the **Predict** tab
   → See the predicted sign + confidence gauge.

2. **Preprocess tab**
   → View the grayscale equalized 32×32 input.

3. **Top-K tab**
   → Explore the top 5 predictions.

4. **Chatbot panel** (right side)
   - Type a question and click **Send**
   - Example: "What does Yield mean?" or "Why is my confidence low?"

---

## 📂 Project Structure

```
your_project/
├── app.py        # Streamlit app
├── model.h5      # Trained traffic sign recognition model
└── README.md     # This file
```
