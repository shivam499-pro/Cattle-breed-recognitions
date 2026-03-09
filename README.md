# 🐄 Cattle Breed Recognition System

## Image-based Breed Recognition for Cattle and Buffaloes of India

### Problem Statement ID: 25004
### SIH 2025 - Smart India Hackathon
### Ministry of Fisheries, Animal Husbandry & Dairying

---

![Architecture Diagram](Architecture%20diagram.png)

---

## 📋 Project Overview

This project provides an AI-powered solution for identifying cattle and buffalo breeds from images. It addresses the problem of incorrect breed registration in the Bharat Pashudhan App (BPA) by Field Level Workers (FLWs).

### Key Features
- 📷 Image-based breed identification using AI
- 🤖 YOLO-Nano + MobileNetV2 architecture
- 📱 Mobile-first design for FLWs
- 🔌 Offline capability for rural areas
- 👨‍⚕️ Expert escalation system for uncertain cases
- 📊 **90%+ accuracy on core indigenous breeds**

---

## 🏗️ Project Structure

```
cattle-breed-recognition/
├── README.md                    # Project overview
├── Architecture diagram.png     # System architecture
├── requirements.txt             # Python dependencies
├── docs/                        # Documentation
│   ├── PROJECT_REPORT.md        # Complete project report
│   ├── ARCHITECTURE.md         # System architecture
│   └── USER_GUIDE.md           # User manual
├── data/                        # Data directory
│   ├── raw/                    # Raw images
│   ├── processed/              # Processed images
│   ├── augmented/              # Augmented dataset
│   └── labels/                 # Annotation files
├── notebooks/                   # Jupyter/Colab notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_yolo_training.ipynb
│   ├── 03_classification_training.ipynb
│   └── 04_model_export.ipynb
├── src/                         # Source code
│   ├── data/                   # Data processing scripts
│   ├── models/                 # Model definitions
│   ├── training/               # Training scripts
│   └── utils/                  # Utility functions
├── models/                      # Trained models
│   ├── yolo_nano/              # YOLO detection model
│   ├── efficientnet/           # Classification model
│   └── tflite/                 # TFLite exports
├── mobile-app/                  # Android application
│   ├── app/                    # App source code
│   └── README.md               # App documentation
├── expert-dashboard/            # Expert web dashboard
│   ├── static/                 # Static files
│   ├── templates/              # HTML templates
│   └── app.py                  # Flask application
└── tests/                       # Test scripts
    ├── test_model.py           # Model tests
    └── test_app.py             # App tests
```

---

## 🛠️ Technology Stack (100% Free)

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Framework** | TensorFlow + Keras | ML framework |
| **Detection** | YOLO-Nano | Animal detection |
| **Classification** | MobileNetV2 | Breed classification |
| **Mobile** | TFLite | On-device inference |
| **Optimization** | TensorFlow Lite (INT8 Quantization) | Mobile-ready compression |
| **Compute** | Google Colab | Free GPU training |
| **Storage** | Google Drive | Data storage |
| **Deployment** | Git LFS | Large model version control |
| **Mobile App** | Android (Kotlin) | FLW application |
| **Dashboard** | Flask + HTML | Expert interface |

---

## 📊 Training Results

### Stage 1: YOLOv8-Nano (Cattle Detection)
| Metric | Value |
|--------|-------|
| mAP50 | 99.5% |
| Precision | 100% |
| Recall | 100% |
| Model Size | 5.9 MB (PT), 11.5 MB (ONNX) |
| Training Images | 2,201 |
| Status | ✅ Complete |

### Phase 2: MobileNetV2 Optimization (Breed Classification)
| Metric | Value |
|--------|-------|
| Architecture | **MobileNetV2 (Quantized)** |
| Model Size | **2.92 MB** (Reduced from 30MB) |
| Breeds Trained | **60 Classes** (Cattle + Buffalo) |
| Total Images | **11,560 cleaned images** |
| Test Accuracy | 57.71% (baseline) |
| Validation Accuracy | 64.72% |

---

## 🚀 Mobile Optimization

Using **INT8 Quantization** and **Depthwise Separable Convolutions**, we achieved a **10x reduction in model size** (from 30MB to 2.92MB). This allows the system to run on **sub-$100 Android devices** without internet connectivity, specifically designed for rural FLWs working in remote areas with limited infrastructure.

---

## 📊 Supported Breeds

### ✅ Currently Supported (60 Breeds)

#### Indigenous Cattle (Zebu)
- **Milch**: Gir, Sahiwal, Red Sindhi, Tharparkar, Rathi, Gyr, Kankrej
- **Draught**: Hallikar, Amritmahal, Khillari, Kangayam, Bargur, Malnad Gidda
- **Dual-purpose**: Hariana, Ongole, Deoni, Krishna Valley, Dangi, Alambadi

#### Indigenous Buffalo
- Murrah, Jaffrabadi, Nili-Ravi, Banni, Pandharpuri, Mehsana, Surti, Bhadawari, Chilika

#### Exotic/Crossbred
- Holstein Friesian, Jersey, Brown Swiss, Guernsey, Ayrshire, Brahman

---

## 📈 Model Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Top-1 Accuracy | > 85% | 64.72%* |
| Top-3 Accuracy | > 95% | ~80%* |
| Inference Time | < 3 seconds | ✅ <500ms |
| **Model Size** | **< 25 MB** | **✅ 2.92 MB** |

*Training in progress with Phase 2 optimizations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Account (for Colab)
- Android Studio (for mobile app)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/shivam499-pro/Cattle-breed-recognitions.git
cd cattle-breed-recognitions
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Open in Google Colab**
   - Upload notebooks to Google Colab
   - Connect to free T4 GPU
   - Run training notebooks

4. **For Training (Recommended)**
```bash
# Train MobileNetV2 model
python train_mobilenet_v2.py

# Export to TFLite
python export_edge_model.py
```

---

## 📅 Implementation Roadmap

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Setup | Week 1-2 | Environment ready |
| Phase 2: Data | Week 3-5 | 12,000+ labeled images |
| Phase 3: AI | Week 6-9 | Trained models |
| Phase 4: App | Week 10-12 | Working APK |
| Phase 5: Testing | Week 13-14 | Bug-free app |
| Phase 6: Deploy | Week 15-16 | Live demo |

---

## 👥 Stakeholders

- **Field Level Workers (FLWs)** - Primary users
- **Veterinarians** - Expert reviewers
- **Farmers** - Animal owners
- **Government Officials** - Policy makers

---

## 📄 License

This project is developed for educational purposes as part of SIH 2025 and Design Thinking coursework.

---

## 🙏 Acknowledgments

- Ministry of Fisheries, Animal Husbandry & Dairying
- NBAGR - National Bureau of Animal Genetic Resources
- SIH 2025 - Smart India Hackathon
