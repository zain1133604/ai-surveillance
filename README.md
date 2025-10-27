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

### 👤 Human Detection
| Model | Precision | Recall | mAP50 | mAP50-95 |
|--------|------------|---------|--------|-----------|
| YOLOv8-L (Teacher) | 0.8160 | 0.7191 | 0.8076 | 0.6016 |
| YOLOv8-Nano (Student) | 0.7904 | 0.6491 | 0.7357 | 0.5078 |

### 🔫 Weapon Detection
| Model | Precision | Recall | mAP50 | mAP50-95 |
|--------|------------|---------|--------|-----------|
| YOLOv8-L (Teacher) | 0.9102 | 0.90383 | 0.9421 | 0.7110 |
| YOLOv8-Small (Student) | 0.8860 | 0.8546 | 0.9097 | 0.6630 |

### 🙂 Face Detection
| Model | Precision | Recall | mAP50 | mAP50-95 |
|--------|------------|---------|--------|-----------|
| YOLOv8-M (Teacher) | 0.9025 | 0.6896 | 0.7670 | 0.5063 |
| YOLOv8-Nano (Student) | 0.881 | 0.6125 | 0.693 | 0.4457 |


---

## 🧾 Features Summary


✅ Real-time Human, Weapon, and Face detection using YOLOv8                                                                                                                                                 
✅ Multi-object tracking with ByteTrack (Human IDs)                                                                                                                                                        
✅ Restricted Area alerts with user-drawn ROI                                                                                                                                                            
✅ Missing person recognition using FaceNet + Face Alignment                                                                                                                                               
✅ Gradio-based web app for interactive video processing                                                                                                                                                  

---






