```markdown
# YOLOv8 Custom Object Detection Project

This project demonstrates training and testing a **custom YOLOv8 object detection model** using a self-prepared dataset.  
The goal was to understand the **complete pipeline** — from environment setup and dataset preparation to training, debugging, and real-time inference using a webcam.

---

## 📌 Project Overview

- Framework: **YOLOv8 (Ultralytics)**
- Task: **Custom Object Detection**
- Classes: Helmet, Jacket
- Training type: Supervised Learning 
- Inference: Image, Video, and Live Camera
- Hardware: NVIDIA GPU (CUDA supported)

This project focuses on **hands-on learning**, including fixing real errors such as dataset path issues, YAML syntax problems, and GPU configuration.

---

## 🗂 Project Structure

```

project/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data_custom.yaml
├── runs/
│   └── detect/
├── train.py
└── README.md

````

---

## ⚙️ Environment Setup

Download the Drivers through the following repositary for Cuda and Cudnn
https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning

Use Anaconda Prompt 
### 1️⃣ Create Virtual Environment

make a project folder ex. named yolo_project


conda create yolo_project -m python=3.9 -y

conda activate yolo_project


### 2️⃣ Install Required Libraries

pip install ultralytics opencv-python

### 3️⃣ Install PyTorch (CUDA 11.8 recommended)

uninstall if already exist

pip uninstall torch torchvision torchaudio -y

now install 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify GPU:

```python
import torch
torch.cuda.is_available()
```
if cuda is not available then download correct version through Pytorch
---

## 📝 Dataset Configuration

write data_custom.yaml


## 🚀 Model Training

Recommended model for beginners and small datasets:

```bash
yolo detect train model=yolov8n.pt data=data_custom.yaml epochs=50 imgsz=640 batch=8 device=0
```

Training results are saved in:

```
runs/detect/train/weights/best.pt
```

---

rename it to yolov8_custom

## 🧪 Testing the Model

### 📷 Image

```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=image.jpg      
```

### 🎥 Video

```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=v1.mp4   
```

### 📸 Live Camera

```bash
yolo detect predict model=yolov8m_custom.pt source=0 show=True conf=0.4 device=0
```

Press **Q** or **Ctrl + C** to stop the camera.

---

## 🧠 Key Learnings

* Correct environment setup is critical for GPU usage
* YAML files are strict — small syntax errors can stop training
* Dataset structure must match YOLO expectations
* Real-time inference is the most motivating part of the project
* Debugging is where real learning happens

---

## 🔮 Future Improvements

* Increase dataset size for better accuracy
* Try YOLOv8s / YOLOv8m for comparison
* Improve FPS using TensorRT or ONNX
* Deploy as a desktop or web application

---
