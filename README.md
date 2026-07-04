# PART·SCAN — Mechanical Parts Classifier

Streamlit app (Vietnamese UI) that classifies **bearing / bolt / gear / nut** using two models trained on the same dataset, plus a Gemini-powered AI assistant:

| Model | File | What it does |
|---|---|---|
| CNN | `mechanical_parts_cnn_final.tflite` | Classifies the whole image (4-class softmax, 128×128 input) |
| YOLOv8 | `best.pt` | Detects each object in the image with a bounding box and class |

## Requirements

- **Python 3.11** (TensorFlow does not support Python 3.14 yet)
- Install dependencies:

```powershell
py -3.11 -m pip install -r requirements.txt
```

## Setup: AI assistant key

The "Trợ lý AI" chatbot needs a free Google AI Studio API key:

1. Get a key at https://aistudio.google.com/apikey
2. Open `.env` in this folder and paste it: `GEMINI_API_KEY=your_key_here`
3. Restart the app.

Without a key the rest of the app works normally; only the chatbot page shows a setup notice.

## Run

```powershell
py -3.11 -m streamlit run app.py
```

The app opens at http://localhost:8501.

## Features

- **Engine selector** (sidebar): CNN, YOLOv8, or Both side-by-side for comparison.
- **Three input modes** (all work with every engine):
  - **Tải ảnh (Upload)** — one or more images (JPG/PNG/WEBP/BMP).
  - **Chụp ảnh (Take Photo)** — webcam snapshot via the browser.
  - **Trực tiếp (Live)** — continuous real-time video classification/detection (streamlit-webrtc); YOLO boxes render per frame and the CNN verdict is overlaid on the stream.
- **Confidence charts** — per-class probability bars for the CNN, per-detection confidence table for YOLO.
- **Downloads** — annotated detection image (PNG) and prediction report (JSON).
- **Thông tin Model page** — architecture datasheets for both models.
- **Trợ lý AI page** — Gemini chatbot (`gemini-2.5-flash`) that answers questions about this website and mechanical equipment only; it politely refuses other topics. Replies in Vietnamese by default.

## Deployment (Streamlit Community Cloud)

The repo is deploy-ready for [share.streamlit.io](https://share.streamlit.io):

1. New app → pick this repo, branch `main`, main file `app.py`.
2. Advanced settings → Python **3.11**.
3. Secrets → add `GEMINI_API_KEY = "your_key_here"` to enable the chatbot (optional; the rest of the app works without it).

`packages.txt` provides the OpenCV system libraries. The CNN ships as a TFLite conversion (`.tflite`, identical predictions) running on the lightweight LiteRT runtime, because full TensorFlow segfaults on Streamlit Cloud's container; the original Keras checkpoints stay local (gitignored).

## Notes

- UI text is Vietnamese with technical terms kept in English; part classes display as "Vòng bi (Bearing)" etc. Labels drawn on detection images stay English.
- The CNN's class order is assumed alphabetical (`Bearing, Bolt, Gear, Nut`). If your training used a different order, edit `CNN_CLASSES` at the top of `app.py`.
- YOLO runs at `conf ≥ 0.25`, `imgsz 800` (640 in live mode). Adjust the constants at the top of `app.py` if needed.
- Do not commit `.env` to version control — it contains your API key.
