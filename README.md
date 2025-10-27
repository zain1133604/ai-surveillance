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
## Test link:

https://huggingface.co/spaces/zain1133604/AI_Surveillance

---
## 🚀 Project Overview

An AI-powered real-time surveillance system integrating Human Detection, Weapon Detection, Restricted Area Monitoring, and Missing Person Recognition. Processes videos and identifies security breaches automatically: 

### 🧍 1. Human Detection & Tracking
- YOLOv8 Human Detector (Medium to Nano): trained on ~50,000 human images.
- Detects and tracks humans in real-time using **YOLOv8 (distilled lightweight model)**.
- Each person is assigned a **unique ID** using **ByteTrack**.

### 🔫 2. Weapon Detection (Gun & Knife) 
- YOLOv8 Weapon Detector (Large to Small): trained on ~120,000 weapon images (guns & knives).
- Detects weapons using a **custom YOLOv8-Small model** distilled from a 120k weapon dataset.
- Automatically raises red alerts when a weapon is detected.

### 🚷 3. Restricted Area Monitoring 
- User paints a **restricted zone** directly on the video’s first frame (using Gradio ImageEditor).
- System monitors entry of humans into that region in real-time.
- Intrusion triggers visual and textual alerts in the final output video.

### 🕵️ 4. Missing Person Recognition
- YOLOv8 Face Detector (Medium to Nano): trained on ~40,000 face images.
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
| **Deployment** | Hugging Face Space (with gradio) |

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

| Model | Teacher | Student |  Images | Results |
|--------|----------|----------|----------|---------|
| YOLOv8-L | → YOLOv8-Nano | Human Detection | 50,000 |
| YOLOv8-L | → YOLOv8-Small | Weapon Detection | 120,000 |
| YOLOv8-M | → YOLOv8-Nano | Face Detection | 40,000 |

---

## 🧾 Features Summary


✅ Real-time Human, Weapon, and Face detection using YOLOv8                                                                                                                                                 
✅ Multi-object tracking with ByteTrack (Human IDs)                                                                                                                                                        
✅ Restricted Area alerts with user-drawn ROI                                                                                                                                                            
✅ Missing person recognition using FaceNet + Face Alignment                                                                                                                                               
✅ Gradio-based web app for interactive video processing                                                                                                                                                  

---






