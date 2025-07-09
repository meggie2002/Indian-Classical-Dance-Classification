# Indian Classical Dance Classification Using Deep Learning 

This project aims to classify images of **Indian classical dance forms** using **Convolutional Neural Networks (CNN)** and **transfer learning techniques** (VGG16 and InceptionV3). The dataset includes 8 traditional Indian dance styles, and the model predicts the correct class from a given image.

---

## Dataset Overview

**Source**: [Kaggle - Indian Dance Form Recognition Dataset](https://www.kaggle.com/datasets/somnath796/indian-dance-form-recognition)
- **Total Images**: 520  
- **Classes**:
  - Bharatanatyam
  - Kathak
  - Kathakali
  - Kuchipudi
  - Manipuri
  - Mohiniyattam
  - Odissi
  - Sattriya
- **Train/Test Split**: 80/20

Images were cleaned and filtered to ensure only available files were used. Stratified sampling was applied for balanced class representation.

---

## Project Pipeline

### 1. Data Cleaning & Preprocessing
- Combined and filtered `train.csv` and `test.csv`
- Verified presence of all listed image files
- Visualized class distribution
- Applied `ImageDataGenerator` with:
  - Rescaling
  - Random rotation, zoom, flips, shifts

### 2. Models Implemented

#### Custom CNN
- 4 Conv layers + MaxPooling
- Fully connected + Dropout
- Softmax output layer
- **Accuracy**: ~38%

####  VGG16 (Transfer Learning)
- Fine-tuned last 4 layers
- Pretrained weights from ImageNet
- **Accuracy**: ~48%

#### InceptionV3 (Transfer Learning)
- Frozen base model
- Custom top layers + Dropout
- **Accuracy**: **~63%**

---

## Model Performance

| Model       | Train Accuracy | Val Accuracy | Test Accuracy |
|-------------|----------------|--------------|----------------|
| CNN         | 42%            | ~39%         | ~38%           |
| VGG16       | 62%            | ~44%         | ~48%           |
| InceptionV3 | 72%            | **~66%**     | **~63%**       |

---

## Evaluation Metrics

### Confusion Matrix – InceptionV3

![Confusion Matrix](https://github.com/user-attachments/assets/7aadedbd-8160-4a1f-bbc7-7d10d5037db9)


### Classification Report – InceptionV3

| Class         | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Bharatanatyam | 0.46      | 0.60   | 0.52     |
| Kathak        | 0.78      | 0.78   | 0.78     |
| Kathakali     | 0.82      | 1.00   | 0.90     |
| Kuchipudi     | 0.60      | 0.33   | 0.43     |
| Manipuri      | 0.80      | 0.57   | 0.67     |
| Mohiniyattam  | 0.57      | 0.80   | 0.67     |
| Odissi        | 0.58      | 0.70   | 0.64     |
| Sattriya      | 0.50      | 0.22   | 0.31     |

---

## Highlights

- Used **EarlyStopping** to prevent overfitting
- Applied **class weighting** to handle imbalance
- Integrated **transfer learning** for better feature extraction
- Achieved **+25% improvement** over base CNN using InceptionV3

---

## Tech Stack

| Area               | Tools Used                                  |
|--------------------|---------------------------------------------|
| Language           | Python                                      |
| Deep Learning      | TensorFlow, Keras                           |
| Data Visualization | Matplotlib, Seaborn                         |
| Image Handling     | Pillow, OpenCV                              |
| Data Processing    | Pandas, NumPy                               |
| Notebook           | Google Colab                                |
| Transfer Learning  | VGG16, InceptionV3 (ImageNet pretrained)    |

---
### Limitations
- Small dataset size (only 520 images)
- High visual similarity among classes (e.g., costumes, poses)
- Some classes like **Odissi** underrepresented

### Future Enhancements
- Collect a larger and more diverse dataset
- Apply advanced models like **EfficientNet** or **Vision Transformers**
- Implement ensemble techniques
- Add explainability using **Grad-CAM** or **LIME**


## Acknowledgements
- Kaggle Dataset - Indian Dance Form Recognition
- TensorFlow & Keras Documentation
- Google Colab for GPU Resources
