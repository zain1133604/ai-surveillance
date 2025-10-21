# ai-surveillance
Built a full Intelligent monitoring system capable of detecting humans, weapons, restricted area breaches, and missing persons in real time. Integrated YOLOv8 with ByteTrack for tracking, and FaceNet for face recognition. Designed a user-friendly Gradio UI for real-time alert visualization and annotation.


# 🧠 Real-Time Restricted Area Monitoring & Missing Person Detection System

An **AI-powered real-time surveillance system** that integrates **Human Detection**, **Weapon Detection**, **Restricted Area Intrusion Monitoring**, and **Missing Person Identification** — all in one intelligent video analytics application.

Built using **YOLOv8**, **ByteTrack**, and **FaceNet**, this system processes uploaded videos (or live streams) and identifies security breaches and missing persons automatically.

<div align="center">
  <img src="https://img.shields.io/badge/Framework-Gradio-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Backend-PyTorch-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Models-YOLOv8%20%7C%20FaceNet%20%7C%20ByteTrack-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Deployed%20On-Hugging%20Face-yellow?style=for-the-badge">
</div>

---

## 🚀 Project Overview

This application combines **three independent AI systems** into a single intelligent video monitoring tool:

### 🧍 1. Human Detection & Tracking
- Detects and tracks humans in real-time using **YOLOv8 (distilled lightweight model)**.
- Each person is assigned a **unique ID** using **ByteTrack**.

### 🔫 2. Weapon Detection (Gun & Knife)
- Detects weapons using a **custom YOLOv8-Small model** distilled from a 115k weapon dataset.
- Automatically raises red alerts when a weapon is detected.

### 🚷 3. Restricted Area Monitoring
- User paints a **restricted zone** directly on the video’s first frame (using Gradio ImageEditor).
- System monitors entry of humans into that region in real-time.
- Intrusion triggers visual and textual alerts in the final output video.

### 🕵️ 4. Missing Person Recognition
- Upload a **reference image** of the missing person.
- The system uses **Face Detection (YOLOv8)** and **Face Recognition (FaceNet + Face Alignment)** to locate them in the video.
- Displays a bold yellow **“MATCH FOUND!”** overlay and highlights the identified person.

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.10 |
| **Framework** | Gradio (UI) |
| **Core AI Models** | YOLOv8 (Ultralytics), ByteTrack, FaceNet |
| **Face Alignment** | face-alignment library |
| **Model Compression** | Knowledge Distillation |
| **Deployment** | Hugging Face Space (with ONNX optimized models) |

---

## ⚙️ Architecture
Architecture Overview – Intelligent Video Surveillance App

1. User Input:
   - Upload a video.
   - Select mode: normal, restricted, or missing_person.
   - Optionally draw Restricted Area or upload reference face image.

2. Preprocessing Module:
   - Extract first frame from video.
   - Generate ROI mask for Restricted Area mode.
   - Preprocess reference face for FaceNet embedding.

3. Detection Module (YOLOv8):
   - Human Detector (Nano) - detect & track humans.
   - Weapon Detector (Small) - detect guns/knives.
   - Face Detector (Nano) - detect faces for missing person recognition.

4. Tracking Module (ByteTrack):
   - Assign unique IDs to humans.
   - Track humans frame by frame.

5. Face Recognition Module (FaceNet + Face Alignment):
   - Align detected faces.
   - Match with reference image to detect missing persons.

6. Alert & Annotation Module:
   - Draw bounding boxes and labels.
   - Flag Restricted Area breaches.
   - Highlight detected weapons and missing persons.
   - Output annotated video with alerts.


---

## 🧠 Distillation Summary

| Model | Teacher | Student | Purpose |
|--------|----------|----------|----------|
| YOLOv8-L | → YOLOv8-Nano | Human Detection |
| YOLOv8-L | → YOLOv8-Small | Weapon Detection |
| YOLOv8-M | → YOLOv8-Nano | Face Detection |

---

## 🧾 Features Summary

✅ Real-time Human + Weapon detection  
✅ Region-based intrusion alerts  
✅ Missing person recognition with FaceNet  
✅ Multi-object tracking with ByteTrack  
✅ Deployed on Hugging Face for testing  
✅ ONNX-optimized models for faster inference  

---






