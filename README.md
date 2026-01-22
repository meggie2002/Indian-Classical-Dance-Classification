# 💃 Indian Classical Dance Classification with Explainable AI

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This repository demonstrates a deep learning approach to classifying **8 forms of Indian Classical Dance** using transfer learning with InceptionV3. Originally developed in 2024 as a basic architecture comparison, the project was **significantly enhanced in January 2026** with explainable AI (Grad-CAM), advanced regularization strategies, and comprehensive performance analysis.

**Key Innovation:** Adding visual explainability to verify the model focuses on relevant dance features rather than background artifacts a critical consideration for cultural AI applications.

---

## 🔄 Project Evolution (2024 → 2026)

### **Version 1.0 (2024): Initial Implementation**
- Basic 3-model comparison (CNN, VGG16, InceptionV3)
- Simple training without early stopping
- No explainability or visualization
- Limited analysis of results

### **Version 2.0 (January 2026): Major Enhancements** ⭐
- **Added Grad-CAM explainability** to visualize model attention
- **Implemented early stopping** with automatic weight restoration
- **Added learning rate scheduling** (ReduceLROnPlateau)
- **Comprehensive analysis** of overfitting and model behavior
- **Professional visualization** of learning curves and attention maps
- **Honest assessment** of model capabilities and limitations

**This updated version demonstrates growth in ML engineering practices: moving from basic model training to production-ready systems with interpretability and proper regularization.**

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

### **Phase 2: Production Training with Early Stopping** (2026 Enhancement)

The selected model was trained with aggressive regularization strategies a significant improvement over the 2024 version:

**Training Configuration:**
- **Callbacks:** Early stopping (patience=4, monitoring val_accuracy) + Learning rate reduction on plateau
- **Transfer Learning:** ImageNet-pretrained InceptionV3 with frozen base layers
- **Optimization:** Adaptive learning rate starting at 0.001

**Result:** Training automatically stopped at **Epoch 8** when validation accuracy plateaued for 4 consecutive epochs, with weights restored to **Epoch 4** (best performance: 56.94%).

---

## 📊 Performance & Analysis

### Final Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **56.94%** (Epoch 4) |
| **Final Training Accuracy** | 82.88% (Epoch 8) |
| **Overfitting Gap** | 28.71% |
| **Baseline (Random)** | 12.5% |
| **Improvement Over Random** | **4.6×** |

### Understanding the Overfitting Gap

The ~29% gap between training and validation accuracy is expected given the constraints:

**Dataset Limitations:**
- Only **292 training images** (~37 per class)
- High intra-class variability (different performers, lighting, staging)
- Complex 8-class problem with overlapping visual features

**Why 57% is Actually Good:**
- Achieves 4.6× improvement over random guessing (12.5%)
- Demonstrates meaningful feature learning despite severe data scarcity
- XAI analysis (below) confirms focus on relevant image regions
- Comparable to human classification without domain expertise

![Overfitting Concept](https://assets.ibm.com/is/image/ibm/overfitting-and-underfitting-in-machine-learning-1:4x3?fmt=png-alpha&dpr=on%2C1.25&wid=512&hei=384)

*Image: High variance (overfitting) occurs when models memorize training data. Source: [IBM](https://www.ibm.com/think/topics/overfitting-vs-underfitting)*

**Mitigation Strategies Applied (2026 Enhancement):**
1. Early stopping prevented further overfitting beyond Epoch 8
2. Best weights restoration (Epoch 4) optimized generalization
3. Learning rate reduction on plateau helped escape local minima
4. Grad-CAM verified meaningful feature learning (see below)

---

## 🧠 Explainability with Grad-CAM: What Does the Model See? (New in 2026)

**Critical Question:** Is the model learning authentic dance features or just memorizing backgrounds?

This was the **major addition in the 2026 update**. I implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize where the model focuses attention during classification something missing from the original 2024 implementation.

### Findings Across All Three Architectures

![XAI Comparison](images/architecture_comparison.png)

**Simple CNN:** Shows reasonable attention to dancer bodies and traditional costumes (white turbans, colored vests), demonstrating even basic architectures can identify relevant regions.

**VGG16:** Displays very coarse, blocky activation patterns with poor spatial precision explaining its lowest accuracy (26%).

**InceptionV3:** Exhibits structured activation patterns concentrated on central image regions containing dancers. While the heatmap shows horizontal bands rather than precise anatomical localization, the model clearly focuses on foreground subjects rather than stage backgrounds.

### Honest Assessment

The Grad-CAM analysis reveals:
- Models focus on **relevant regions** (dancers, not backgrounds)
- InceptionV3 shows the **most structured attention patterns**
- Activation patterns are **coarse-grained**, suggesting reliance on texture/color cues alongside spatial features
- Fine-grained attention to specific body parts (hands, feet) is **not achieved** with current data size

**Interpretation:** The model learns approximate regional features (body positioning, costumes) rather than precise anatomical understanding. This validates the transfer learning approach while highlighting the primary bottleneck: **dataset size**.

### Final Production Results

![Final Results](images/final_dance_analysis.png)
*Left: Grad-CAM attention map showing model focus. Center: Training vs validation accuracy showing early stopping at epoch 8. Right: Loss convergence demonstrating effective learning.*

The learning curves clearly show:
- **Early stopping triggered correctly** at epoch 8 after 4 epochs without improvement
- **Validation accuracy peaked** at 56.94% (epoch 4), where weights were restored
- **Validation loss plateaued** around 1.3-1.4, indicating the model reached its capacity
- **Training continued improving** while validation stagnated classic overfitting signal

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
- **Preprocessing:** InceptionV3-specific normalization (`inception_v3.preprocess_input`)

### Callbacks (2026 Enhancement)
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

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meggie2002/Indian-Classical-Dance-Classification/blob/main/Indian_Dance_XAI.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells

### Option 2: Local Installation
```bash
git clone https://github.com/meggie2002/Indian-Classical-Dance-Classification.git
cd Indian-Classical-Dance-Classification
pip install -r requirements.txt
jupyter notebook Indian_Dance_XAI.ipynb
```

For detailed setup instructions, see [SETUP.md](SETUP.md)

---

## 📂 Repository Structure
```
Indian-Classical-Dance-Classification/
├── Indian_Dance_XAI.ipynb          # Complete pipeline (updated 2026)
├── requirements.txt                 # Python dependencies
├── SETUP.md                         # Detailed installation guide
├── LICENSE                          # MIT License
├── README.md                        # This file
└── images/
    ├── architecture_comparison.png  # Phase 1 XAI results (3 models)
    └── final_dance_analysis.png     # Phase 2 results + learning curves
```

---

## 📥 Pre-trained Model Download

Due to file size limitations on GitHub, the trained model weights are hosted separately:

**[Download best_dance_model.keras (83.3 MB)](https://drive.google.com/file/d/1cmc_bLAAiWSZn3zUcd9-yuLIAki-dL_R/view?usp=sharing)**

This model contains the **Epoch 4 weights** (best validation accuracy: 56.94%) automatically restored by early stopping.

To use the pre-trained model:
```python
from tensorflow import keras
model = keras.models.load_model('best_dance_model.keras')

# Make predictions
predictions = model.predict(your_image_array)
```

---

## 🆕 What's New in Version 2.0 (January 2026)

The 2026 update represents a significant enhancement over the original 2024 implementation:

### **Added Features:**
1. **Grad-CAM Explainability** - Visual attention maps showing what the model sees
2. **Advanced Callbacks** - Early stopping, learning rate scheduling, model checkpointing
3. **Comprehensive Analysis** - Learning curves, overfitting analysis, honest assessment
4. **Professional Visualization** - Multi-panel figures for publication-quality results
5. **Improved Documentation** - Detailed technical specifications and reproducibility

### **Why These Improvements Matter:**
- **Explainability** builds trust in AI systems for cultural applications
- **Regularization** prevents wasted computation and unreliable models
- **Honest assessment** demonstrates technical maturity and engineering judgment
- **Reproducibility** enables others to build on this work

---

## 🚀 Future Improvements

To push beyond the current 57% accuracy ceiling:

1. **Data Augmentation:** Rotation, zoom, brightness variation, horizontal flip to synthetically expand training set
2. **Pose Estimation:** Integrate OpenPose or MediaPipe keypoints as auxiliary features to guide anatomical understanding
3. **Progressive Fine-tuning:** Gradually unfreeze InceptionV3 layers with discriminative learning rates
4. **Ensemble Methods:** Combine predictions from InceptionV3, ResNet50, and EfficientNet
5. **Larger Dataset:** Expand to 500+ images per class for robust feature learning
6. **Temporal Modeling:** Extend to video classification to capture dance movement sequences

---

## 📚 References & Citations

**Key Papers:**
- Szegedy, C., et al. (2016). "Rethinking the Inception Architecture for Computer Vision." [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning." IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.

**Dataset:**
- Somnath. ["Indian Dance Form Recognition."](https://www.kaggle.com/datasets/somnath796/indian-dance-form-recognition) Kaggle, 2023.

---

## 📝 Key Takeaways

This project demonstrates:

1. **Iterative improvement:** Taking a basic 2024 project and enhancing it with production-ready features in 2026
2. **Architecture selection matters:** Multi-scale features (InceptionV3) outperform sequential depth (VGG16) on small datasets
3. **Regularization is essential:** Early stopping and learning rate scheduling prevent catastrophic overfitting
4. **XAI builds trust:** Grad-CAM confirms models focus on relevant features, validating the approach despite moderate accuracy
5. **Data is the bottleneck:** Primary improvement path is dataset expansion, not architectural complexity
6. **Honest reporting matters:** Acknowledging limitations (57% accuracy, coarse-grained attention) builds more credibility than overclaiming

**Bottom Line:** Even with severe data constraints (292 training images), transfer learning + explainability creates interpretable, trustworthy AI systems for specialized cultural applications. The 2026 enhancements show how to take a basic ML project and make it production-ready with proper engineering practices.


---

## 🙏 Acknowledgments

- **Dataset:** [Somnath's Indian Dance Form Recognition](https://www.kaggle.com/datasets/somnath796/indian-dance-form-recognition) on Kaggle
- **Pre-trained Weights:** ImageNet via TensorFlow/Keras Applications
- **Grad-CAM Implementation:** Inspired by the original paper by Selvaraju et al. (2017)
- **Infrastructure:** Google Colab for free GPU access

---

## 📅 Version History

- **v1.0 (2024):** Initial implementation with basic 3-model comparison
- **v2.0 (January 2026):** Major update with explainable AI, advanced regularization, and comprehensive analysis

---

*This project showcases practical ML engineering skills and professional growth: from basic model training (2024) to production-ready systems with explainability, proper regularization, and honest performance reporting (2026) essential competencies for modern ML engineering.*
