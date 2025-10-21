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
flowchart TD
    A[User Input] -->|Upload Video| B[Extract First Frame]
    A -->|Draw Restricted Area| C[Generate ROI Mask]
    A -->|Upload Reference Face| D[Preprocess Reference Face]
    
    B --> E[Detection & Tracking Pipeline]
    C --> E
    D --> E

    subgraph Detection & Tracking Pipeline
        E1[Human Detection & ByteTrack] --> E2{Check Restricted Area?}
        E2 -->|Yes| E3[Alert: Restricted Breach]
        E2 -->|No| E4[Normal ID Bounding Box]
        
        E5[Weapon Detection] --> E6[Weapon Alerts]
        
        E7[Face Detection + FaceNet] --> E8{Matches Reference?}
        E8 -->|Yes| E9[Yellow Bounding Box + "MATCH FOUND!"]
        E8 -->|No| E10[No Action]
    end

    E --> F[Draw Bounding Boxes & Labels]
    F --> G[Generate Alert Log]
    F --> H[Output Processed Video]

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






