# 💃 Indian Classical Dance Classification & Explainability (XAI)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## 📌 Project Overview
This repository contains a deep learning framework designed to classify **8 forms of Indian Classical Dance** (Bharatanatyam, Kathak, Kathakali, Kuchipudi, Manipuri, Mohiniyattam, Odissi, and Sattriya). 

While many models act as "black boxes," this project implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations for model predictions. This ensures the model identifies dance forms based on authentic cultural markers—like hand gestures (**Mudras**) and leg postures—rather than background artifacts.

---

## 🚀 The Development Story: Benchmarking to Production

### **Phase 1: Architecture Benchmarking**
In the prototyping phase, I compared three distinct architectures over a 3-epoch trial to evaluate feature extraction efficiency:
* **Simple CNN:** Baseline performance; struggled with spatial localization.
* **VGG16:** Strong feature extraction but computationally heavy.
* **InceptionV3:** The clear winner, showing the fastest convergence and most precise "attention" heatmaps.

### **Phase 2: Production & Optimization**
The winning **InceptionV3** model was trained for 25 epochs using Transfer Learning. I implemented a robust training pipeline including:
* **Early Stopping:** Monitoring validation loss to restore weights from the optimal epoch (Epoch 20).
* **Learning Rate Reduction:** Fine-tuning the optimizer as the model approached a local minima.

---

## 📊 Performance Analysis
| Metric | Result |
| :--- | :--- |
| **Best Model** | InceptionV3 (Pre-trained on ImageNet) |
| **Training Accuracy** | **99.42%** |
| **Validation Accuracy** | **66.67%** |
| **Primary Challenge** | High variance/overfitting due to specialized dataset size. |



---

## 🧠 Explainability (XAI) with Grad-CAM
The most critical part of this project is verifying **why** the model makes a decision. Using Grad-CAM, we generated heatmaps that overlay the "focus area" of the neural network onto the original image.

**Key Insight:** Our production model successfully learned to ignore the background and focus on the dancer's silhouette and limb positions, providing a transparent and reliable classification process.



---

## 🛠️ Tech Stack & Tools
* **Deep Learning:** TensorFlow 2.x, Keras 3.0
* **Computer Vision:** OpenCV, PIL
* **Architecture:** InceptionV3 (Multi-scale Feature Extraction)
* **Data Source:** Kaggle (Indian Dance Form Recognition)

---

## 📂 Repository Structure
* `Indian_Dance_XAI.ipynb`: Full end-to-end pipeline (Data loading -> Benchmarking -> Production).
* `best_dance_model.keras`: Saved weights of the optimized InceptionV3 model.
* `final_dance_analysis.png`: Comparison plot of XAI heatmaps and learning curves.
* `leaderboard.csv`: Historical accuracy data from Phase 1 comparison.

---

## 📝 Conclusion
By combining **Transfer Learning** with **Explainable AI**, this project bridges the gap between high-performance computing and cultural preservation. It proves that deep learning models can be trained to respect the technical nuances of classical arts while providing transparent reasoning for their classifications.

---
