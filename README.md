# 🧠 Brain MRI Classification with Diffusion-Based Data Augmentation

## 🚀 Overview

This project explores the use of **diffusion models (DDPM)** for synthetic data generation to improve **medical image classification performance**.

We design and evaluate a full pipeline that:
- Trains a **diffusion model** on brain MRI scans  
- Generates **synthetic MRI images from noise**  
- Uses them for **data augmentation**  
- Measures their impact on a **tumor classification model**

---

## 🎯 Problem Statement

Medical datasets are often:
- Limited in size  
- Imbalanced across classes  
- Expensive to collect and annotate  

This project investigates:

> ❓ *Can diffusion-based synthetic data improve classification performance in medical imaging?*

---

## 🧠 Approach

### 1️⃣ Baseline Classifier
- Model: **ResNet18 (pretrained)**
- Input: Grayscale MRI images (128×128)
- Classes:
  - Glioma  
  - Meningioma  
  - Pituitary  
  - No Tumor  

---

### 2️⃣ Diffusion Model
- Architecture: **Lightweight UNet**
- Method: **Denoising Diffusion Probabilistic Model (DDPM)**
- Trained to:

Learn noise distribution → reconstruct MRI images


- Hardware constraint:
- GPU: **NVIDIA MX450 (2GB VRAM)**
- Optimizations:
  - Reduced timesteps
  - Lightweight architecture
  - Mixed GPU/CPU memory handling

---

### 3️⃣ Synthetic Data Generation
- Generated MRI-like images from random noise  
- Stored in structured dataset:


synthetic/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/


---

### 4️⃣ Data Augmentation Strategy

Three experimental setups were evaluated:

| Setup | Description |
|------|------------|
| Baseline | Real data only |
| Naive Augmentation | Large synthetic dataset |
| Controlled Augmentation | Limited synthetic data (balanced) |

---

## 📊 Results

### Overall Performance

| Experiment | Accuracy |
|----------|----------|
| Baseline | **92.75%** |
| Naive Augmentation (100/class) | **91.44%** ❌ |
| Controlled Augmentation (30/class) | **93.25%** ✅ |

---

### Advanced Metrics (Best Model)

| Metric | Value |
|------|------|
| Accuracy | **93.25%** |
| AUC-ROC | **~0.97** |
| Macro F1-score | **~0.93** |

---

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|------|----------|--------|----------|
| Glioma | 0.97 | 0.79 | 0.87 |
| Meningioma | 0.87 | 0.96 | 0.91 |
| Pituitary | 0.94 | 1.00 | 0.97 |
| No Tumor | 0.97 | 0.98 | 0.98 |

---

## 🔥 Key Insights

### 1️⃣ Diffusion Augmentation Works
- Controlled synthetic data improved performance from:

92.75% → 93.25%


---

### 2️⃣ More Synthetic Data ≠ Better Performance
- Large synthetic datasets degraded performance:

Over-augmentation → noise injection → reduced generalization


---

### 3️⃣ Class-wise Analysis is Critical
- Glioma recall (~0.79) is lower than other classes  
- Indicates need for:
- Class-aware generation  
- Targeted augmentation  

---

### 4️⃣ Data-Centric AI > Model-Centric AI
> Improving data quality had a measurable impact without changing model architecture.

---

## 🏗️ Project Structure


brain-mri-diffusion/
│
├── data/
│ ├── Training/
│ └── Testing/
│
├── synthetic/
│
├── diffusion/
│ ├── model.py
│ ├── train.py
│ ├── sample.py
│ └── generate_dataset.py
│
├── classifier/
│ ├── train.py
│ └── train_augmented_metrics.py
│
├── utils/
│ ├── dataset.py
│ └── combined_dataset.py
│
├── requirements.txt
└── README.md


---

## ⚙️ Tech Stack

- **PyTorch**
- OpenCV
- NumPy
- Matplotlib
- torchvision
- scikit-learn

---

## 🧪 How to Run

### 1️⃣ Train Diffusion Model
```bash
python diffusion/train.py
```

### 2️⃣ Generate Synthetic Dataset
```bash
python diffusion/generate_dataset.py
```

### 3️⃣ Train Augmented Classifier
```bash
python classifier/train_augmented_metrics.py
```

---

## 📦 Model Weights

Model weights (.pth) are not included in this repository.

You can reproduce results by running the training scripts.

## 🧠 Future Improvements

- Conditional diffusion (class-aware generation)
- Improve glioma recall using targeted augmentation
- Higher resolution MRI synthesis
- Attention-based UNet architectures
- Explainability (Grad-CAM)
- Deployment (Streamlit / API)

🏆 Key Takeaway

This project demonstrates a data-centric AI approach, where controlled synthetic data generation using diffusion models leads to measurable improvements in medical image classification.

📌 Author

Adetya Jamwal
BTech | Computer Science | AI/ML

⭐ If you found this useful, please consider giving it a star!