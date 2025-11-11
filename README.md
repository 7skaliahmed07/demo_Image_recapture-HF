# 📸 Screen Recapture Detection

Web app that detects if an image is original or recaptured (photo of a screen) using deep learning.

## 🚀 Live Demo

**Live App:** [https://screen-recapture-detection.onrender.com](https://screen-recapture-detection.onrender.com)

*Note: Free instance may take 30-50 seconds to start after inactivity.*

## 📋 Features

- Upload any image (JPEG, PNG, etc.)
- AI-powered classification: **Original** vs **Recaptured**
- Confidence scores for predictions
- Clean web interface

## 🛠️ Tech Stack

- **Backend**: FastAPI, TensorFlow Lite
- **Frontend**: HTML/CSS/JavaScript
- **Model**: Custom CNN (15.9MB optimized)
- **Hosting**: Render.com

## 🎯 Usage

1. Visit the live app
2. Upload an image
3. Get instant classification with confidence score

## 🔧 API

- `POST /predict` - Classify images
- `GET /health` - Service status

```bash
curl -X POST -F "file=@image.jpg" https://screen-recapture-detection.onrender.com/predict
```

## 🚀 Run Locally

```bash
git clone https://github.com/7skaliahmed07/demo_Image_recapture-Render.git
cd demo_Image_recapture-Render
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:8000`
