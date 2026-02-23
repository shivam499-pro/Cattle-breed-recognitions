# 🐄 Cattle Breed Recognition System

## Image-based Breed Recognition for Cattle and Buffaloes of India

### Problem Statement ID: 25004
### SIH 2025 - Smart India Hackathon
### Ministry of Fisheries, Animal Husbandry & Dairying

---

## 📋 Project Overview

This project provides an AI-powered solution for identifying cattle and buffalo breeds from images. It addresses the problem of incorrect breed registration in the Bharat Pashudhan App (BPA) by Field Level Workers (FLWs).

### Key Features
- 📷 Image-based breed identification using AI
- 🤖 YOLO-Nano + EfficientNet-B0 architecture
- 📱 Mobile-first design for FLWs
- 🔌 Offline capability for rural areas
- 👨‍⚕️ Expert escalation system for uncertain cases
- 📊 85%+ accuracy target for common breeds

---

## 🏗️ Project Structure

```
cattle-breed-recognition/
├── README.md                    # Project overview
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
| **Classification** | EfficientNet-B0 | Breed classification |
| **Mobile** | TFLite | On-device inference |
| **Compute** | Google Colab | Free GPU training |
| **Storage** | Google Drive | Data storage |
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

### Stage 2: EfficientNet-B0 (Breed Classification)
| Metric | Value |
|--------|-------|
| Test Accuracy | 57.71% |
| Validation Accuracy | 64.72% |
| Model Size | ~30 MB |
| Training Images | 1,506 |
| Validation Images | 428 |
| Test Images | 227 |
| Status | ✅ Complete |

### 12 Breeds Trained
| Breed | Precision | F1-Score |
|-------|-----------|----------|
| brahman | 75% | 67% |
| brahman cross | 45% | 45% |
| cholistani | 65% | 60% |
| cholistani cross | 46% | 53% |
| dhani | 100% | 71% |
| fresian | 83% | 77% |
| fresian cross | 44% | 51% |
| kankarej | 100% | 77% |
| sahiwal | 70% | 76% |
| sahiwal cross | 52% | 49% |
| sibbi | 60% | 62% |
| unidentified (mixed) | 60% | 39% |

---

## 📊 Indian Breeds Supported

### Currently Trained (12 Breeds)
- brahman, brahman cross, cholistani, cholistani cross
- dhani, fresian, fresian cross, kankarej
- sahiwal, sahiwal cross, sibbi, unidentified (mixed)

### Target Breeds (35+)
- **Milch**: Gir, Sahiwal, Red Sindhi, Tharparkar, Rathi
- **Draught**: Hallikar, Amritmahal, Khillari, Kangayam
- **Dual-purpose**: Hariana, Kankrej, Ongole, Deoni
- **Buffalo**: Murrah, Jaffrabadi, Nili-Ravi, Banni, Pandharpuri

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Account (for Colab)
- Android Studio (for mobile app)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/cattle-breed-recognition.git
cd cattle-breed-recognition
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Open in Google Colab**
   - Upload notebooks to Google Colab
   - Connect to free T4 GPU
   - Run training notebooks

---

## 📈 Model Performance Targets

| Metric | Target |
|--------|--------|
| Top-1 Accuracy | > 85% |
| Top-3 Accuracy | > 95% |
| Inference Time | < 3 seconds |
| Model Size | < 25 MB |

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
