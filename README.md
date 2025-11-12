---
title: Screen Recapture Detection
emoji: 📸
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 📸 Screen Recapture Detection

A deep learning web application that detects whether an image is **original** or **recaptured** (e.g., photo of a screen, monitor, or display).

## 🚀 Live Demo

**Try it here:** [https://huggingface.co/spaces/UzerDeveloper07/screen-recapture-detection](https://huggingface.co/spaces/UzerDeveloper07/screen-recapture-detection)

## 🎯 What It Does

- **Upload any image** (JPEG, PNG, etc.)
- **AI-powered classification** detects if the image is:
  - 📷 **Original Image** - Direct digital capture
  - 🖥️ **Recaptured Image** - Photo taken of a screen/monitor
- **Confidence scores** show prediction certainty

## 🛠️ Technical Details

- **Model**: Custom Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras → TensorFlow Lite (optimized)
- **Backend**: FastAPI + Gradio interface
- **Input**: 224×224 RGB images
- **Output**: Binary classification with confidence scores

## 🎮 How to Use

1. **Visit** the live demo link above
2. **Upload** an image file
3. **Wait** for AI analysis (2-3 seconds)
4. **View** results showing:
   - Prediction (Original/Recaptured)
   - Confidence percentage
   - Raw prediction score

## 💡 Use Cases

- **Digital forensics** - Detect screen-captured evidence
- **Content moderation** - Identify reposted screen content
- **Academic integrity** - Detect screenshots of online materials
- **Image authentication** - Verify original vs secondary captures

## 🔧 Model Performance

- **Model Size**: 15.9 MB (TensorFlow Lite optimized)
- **Inference Time**: < 3 seconds
- **Accuracy**: [Add your model's accuracy here if known]
- **Optimized** for web deployment with minimal memory usage
