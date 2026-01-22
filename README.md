# 💃 Indian Classical Dance Classification with Explainable AI

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## 📌 Project Overview
This repository demonstrates a deep learning approach to classifying **8 forms of Indian Classical Dance** using transfer learning with InceptionV3. The project emphasizes **explainability through Grad-CAM**, providing visual evidence of what the model learns despite working with a limited dataset (364 images total).

**Key Challenge:** Building a robust classifier with minimal training data while ensuring the model focuses on relevant features rather than background artifacts.

---

## 🚀 Development Approach: From Benchmarking to Production

### **Phase 1: Architecture Comparison (3 Epochs)**
I evaluated three architectures to identify the most effective feature extractor for this specialized task:

| Architecture | Validation Accuracy | Key Characteristic |
|-------------|-------------------|-------------------|
| Custom CNN | 33% | Baseline; insufficient depth for complex patterns |
| VGG16 | 26% | Deep sequential layers struggled with small dataset |
| **InceptionV3** | **46%** | Multi-scale feature extraction proved most effective |

**Winner:** InceptionV3 demonstrated superior convergence and spatial understanding through its parallel convolution paths (1×1, 3×3, 5×5), making it ideal for capturing dance postures at multiple scales.

### **Phase 2: Production Training with Early Stopping**
The selected model was trained with aggressive regularization strategies:

**Training Configuration:**
- **Callbacks:** Early stopping (patience=4, monitoring val_accuracy) + Learning rate reduction on plateau
- **Transfer Learning:** ImageNet-pretrained InceptionV3 with frozen base layers
- **Optimization:** Adaptive learning rate (0.001 → 0.0005 via ReduceLROnPlateau)

**Result:** Training automatically stopped at **Epoch 8** when validation accuracy plateaued, with weights restored to **Epoch 4** (best performance).

---

## 📊 Performance & Analysis

### Final Results
| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **56.94%** (Epoch 4) |
| **Final Training Accuracy** | 82.88% (Epoch 8) |
| **Overfitting Gap** | ~28% |
| **Baseline (Random)** | 12.5% |
| **Improvement Over Random** | **4.6×** |

### Understanding the Overfitting Gap

The ~28% gap between training and validation accuracy is expected given the constraints:

**Dataset Limitations:**
- Only **292 training images** (~37 per class)
- High intra-class variability (different performers, lighting, staging)
- Complex 8-class problem with overlapping visual features

**Why 57% is Actually Good:**
- Achieves 4.6× improvement over random guessing
- Demonstrates meaningful feature learning despite severe data scarcity
- XAI analysis (below) confirms focus on relevant image regions

![Overfitting Concept](https://assets.ibm.com/is/image/ibm/overfitting-and-underfitting-in-machine-learning-1:4x3?fmt=png-alpha&dpr=on%2C1.25&wid=512&hei=384)
*Image: High variance (overfitting) occurs when models memorize training data. Source: [IBM](https://www.ibm.com/think/topics/overfitting-vs-underfitting)*

**Mitigation Strategies Applied:**
1. ✅ Early stopping prevented further overfitting beyond Epoch 8
2. ✅ Best weights restoration (Epoch 4) optimized generalization
3. ✅ Learning rate reduction helped escape local minima
4. ✅ Grad-CAM verified meaningful feature learning (see below)

---

## 🧠 Explainability with Grad-CAM: What Does the Model See?

**Critical Question:** Is the model learning authentic dance features or just memorizing backgrounds?

To answer this, I implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize where the model focuses attention during classification.

### Findings Across All Three Architectures

![XAI Comparison](final_dance_analysis.png)

**Simple CNN:** Shows reasonable attention to dancer bodies and traditional costumes (white turbans, colored vests), demonstrating even basic architectures can identify relevant regions.

**VGG16:** Displays very coarse, blocky activation patterns with poor spatial precision—explaining its lowest accuracy.

**InceptionV3:** Exhibits structured activation patterns concentrated on central image regions containing dancers. While the heatmap shows horizontal bands rather than precise anatomical localization, the model clearly focuses on foreground subjects rather than stage backgrounds.

### Honest Assessment

The Grad-CAM analysis reveals:
- ✅ Models focus on **relevant regions** (dancers, not backgrounds)
- ✅ InceptionV3 shows the **most structured attention patterns**
- ⚠️ Activation patterns are **coarse-grained**, suggesting reliance on texture/color cues alongside spatial features
- ⚠️ Fine-grained attention to specific body parts (hands, feet) is **not achieved** with current data size

**Interpretation:** The model learns approximate regional features (body positioning, costumes) rather than precise anatomical understanding. This validates the transfer learning approach while highlighting the primary bottleneck: **dataset size**.

---

## 🔬 Technical Implementation Details

### Model Architecture
```python
Input (299×299×3)
  ↓
InceptionV3 Base (frozen, ImageNet weights)
  ↓
GlobalAveragePooling2D
  ↓
Dense(8, softmax)
```

### Training Specifications
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam (initial lr=0.001)
- **Batch Size:** 32
- **Data Split:** 80% train (292 images), 20% validation (72 images)
- **Preprocessing:** InceptionV3-specific normalization

### Callbacks
```python
EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
```

### Hardware & Environment
- **GPU:** NVIDIA T4 (Google Colab)
- **Framework:** TensorFlow 2.15, Keras 3.0
- **Training Time:** ~3 minutes (8 epochs)

---

## 📂 Repository Structure
```
├── Indian_Dance_XAI.ipynb          # Complete pipeline (data → training → XAI)
├── final_dance_analysis.png        # Grad-CAM visualizations + learning curves
└── README.md                       # This file
```

---

## 🚀 Future Improvements

To push beyond the current 57% accuracy ceiling:

1. **Data Augmentation:** Rotation, zoom, horizontal flip to synthetically expand training set
2. **Pose Estimation:** Integrate OpenPose or MediaPipe keypoints as auxiliary features
3. **Progressive Fine-tuning:** Gradually unfreeze InceptionV3 layers with discriminative learning rates
4. **Ensemble Methods:** Combine predictions from InceptionV3, ResNet, and EfficientNet
5. **Larger Dataset:** Expand to 500+ images per class for robust feature learning

---

## 📚 References & Citations

**Key Papers:**
- Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning." IEEE TKDE

**Dataset:**
- Somnath. ["Indian Dance Form Recognition."](https://www.kaggle.com/datasets/somnath796/indian-dance-form-recognition) Kaggle, 2023.

---

## 📝 Key Takeaways

This project demonstrates:
1. **Architecture selection matters:** Multi-scale features (InceptionV3) outperform sequential depth (VGG16) on small datasets
2. **Regularization is essential:** Early stopping and learning rate scheduling prevent catastrophic overfitting
3. **XAI builds trust:** Grad-CAM confirms models focus on relevant features, validating the approach despite moderate accuracy
4. **Data is the bottleneck:** Primary improvement path is dataset expansion, not architectural complexity

**Bottom Line:** Even with severe data constraints, transfer learning + explainability creates interpretable, trustworthy AI systems for specialized cultural applications.

---


---

*This project showcases practical ML engineering skills: architecture comparison, overfitting mitigation, explainable AI, and honest performance reporting—essential for production systems.*
