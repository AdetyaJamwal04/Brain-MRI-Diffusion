# 🧠 Brain MRI Classification with Diffusion-Based Data Augmentation

## 🚀 Overview

This project explores the use of **diffusion models for synthetic data generation** to improve **medical image classification performance**.

We build a complete pipeline that:
- Trains a **diffusion model** on brain MRI scans  
- Generates **synthetic MRI images**  
- Uses them for **data augmentation**  
- Evaluates their impact on a **tumor classification model**

---

## 🎯 Problem Statement

Medical datasets are often:
- Limited in size  
- Imbalanced across classes  
- Expensive to collect and annotate  

This project investigates:

> ❓ *Can diffusion-based synthetic data improve classification performance in medical imaging?*

---



---

## 🧠 Approach

### 1️⃣ Baseline Model
- Model: **ResNet18**
- Input: Brain MRI images (grayscale, 128×128)
- Task: Multi-class classification  
  (`glioma`, `meningioma`, `pituitary`, `no_tumor`)

---

### 2️⃣ Diffusion Model
- Architecture: **Custom lightweight UNet**
- Method: **Denoising Diffusion Probabilistic Model (DDPM)**
- Trained to generate synthetic MRI images from random noise

---

### 3️⃣ Synthetic Data Generation
- Generated **MRI-like images from random noise**
- Stored in structured dataset:

```
synthetic/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
```


---

### 4️⃣ Data Augmentation Strategy

We evaluated three setups:

| Setup | Description |
|------|------------|
| Baseline | Real data only |
| Naive Augmentation | Large synthetic dataset |
| Controlled Augmentation | Limited synthetic data (balanced) |

---

## 📊 Results

| Experiment | Test Accuracy |
|----------|--------------|
| Baseline (Real Data Only) | **92.75%** |
| Naive Augmentation (100/class, early training) | **91.44%** ❌ |
| Naive Augmentation (100/class, longer training) | **93.56%** |
| Controlled Augmentation (30/class) | **94.12%** ✅ |

---

## 🔥 Key Insights

- ✅ Diffusion-based augmentation **can improve performance**
- ❌ Excessive synthetic data **can degrade generalization**
- ⚖️ Controlled augmentation provides **optimal results**

> 📌 **Conclusion:** Quality and balance of synthetic data matter more than quantity.

---

## 🏗️ Project Structure

```
brain-mri-diffusion/
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── synthetic/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
├── diffusion/
│   ├── model.py
│   ├── train.py
│   ├── sample.py
│   ├── generate_dataset.py
│   └── diffusion_model_epoch_*.pth
│
├── classifier/
│   ├── train.py
│   └── train_augmented.py
│
├── utils/
│   ├── dataset.py
│   └── combined_dataset.py
│
├── requirements.txt
├── system_architecture.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **PyTorch**
- OpenCV
- NumPy
- Matplotlib
- torchvision

---

## 📥 Data Setup

**Note:** The `data/` and `synthetic/` directories are not included in this repository (see `.gitignore`). 

### Real Dataset
1. Download brain MRI dataset from [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Extract and organize into the following structure:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Synthetic Dataset
The `synthetic/` directory will be automatically created when you run:
```bash
python diffusion/generate_dataset.py
```

This generates MRI-like images organized by tumor class.

---

## 🧪 How to Run

### 1️⃣ Train Diffusion Model
```bash
python diffusion/train.py
```

### 2️⃣ Generate Synthetic Data
```bash
python diffusion/generate_dataset.py
```

### 3️⃣ Train Classifier (Augmented)
```bash
python classifier/train_augmented.py
```

---

## 📦 Model Weights

Trained model weights (.pth) are not included in this repository.

You can reproduce results by running the training scripts above.

---

## 🧠 Future Improvements

- Conditional diffusion (class-aware generation)
- Higher resolution MRI synthesis
- Advanced architectures (attention-based UNet)
- Explainability (Grad-CAM)
- Deployment (Streamlit / API)

---

## 🏆 Key Takeaway

This project demonstrates a data-centric AI approach, where improving data quality (via diffusion models) leads to measurable gains in model performance.

---

## 📌 Author

**Adetya Jamwal**  
BTech | Computer Science | AI/ML Enthusiast

**Give it a ⭐ on GitHub!**
