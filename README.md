# 🧠 UTKFace Age & Gender Prediction using ResNet50 (PyTorch)

A deep learning project for multi-task facial attribute prediction, estimating age (regression) and gender (classification) from facial images using a ResNet50-based architecture.

This project demonstrates the effectiveness of transfer learning with deep residual networks for extracting high-level facial features and solving multiple tasks simultaneously.

---

## 🚀 Introduction

Facial analysis is a fundamental task in computer vision with applications in:

- Human-computer interaction  
- Surveillance and security  
- Social media analytics  
- Healthcare and demographic studies  

This project uses the UTKFace dataset, a large-scale dataset annotated with age and gender, to train a deep neural network capable of learning meaningful facial representations.

The approach is based on a shared feature extractor (ResNet50) with task-specific prediction heads.

---

## 🧠 Model Architecture

### Backbone: ResNet50

The model is built on ResNet50, a deep convolutional neural network that uses residual connections to enable stable training of deep architectures.

Why ResNet50:

- Enables very deep networks without vanishing gradients  
- Strong feature extraction capability  
- Pretrained on ImageNet (transfer learning)  
- Proven performance in face-related tasks  

---

### Architecture Overview

Input Image (224x224)
        ↓
ResNet50 Backbone (Pretrained)
        ↓
Shared Feature Vector
       / \
      /   \
Age Head   Gender Head

- The original fully connected layer is removed  
- The backbone outputs shared features  

Age Head:
- Fully connected layers  
- Batch normalization + dropout  
- Outputs a continuous value  

Gender Head:
- Fully connected layers  
- Outputs class logits (Male / Female)  

---

## ⚙️ Methodology

### Dataset

- Dataset: UTKFace  
- Labels: Age, Gender  

Preprocessing:

- Resize to 224×224  
- Data augmentation (horizontal flip, rotation)  
- Normalize using ImageNet statistics  

---

### Data Split

- 80% Training  
- 10% Validation  
- 10% Test  

---

### Loss Functions

- Age: L1 Loss (Mean Absolute Error)  
- Gender: CrossEntropy Loss  

Total Loss = Age Loss + Gender Loss

---

### Optimization

- Optimizer: Adam  
- Learning Rate: 1e-4  
- Batch Size: 32  

---

## 📊 Results

- Age MAE: 5.80 years  
- Gender Accuracy: 94.01%  
- Precision: 94.21%  
- Recall: 93.39%  
- F1 Score: 93.79%  

---

## 📁 Project Structure

.
├── data_loader.py  
├── model.py  
├── train_model.py  
├── test.py  
├── evaluation.py  
├── final_results.PNG  
└── README.md  

---

## ▶️ How to Run

Train the model:

python train_model.py

Test the model:

python test.py

---

## 🧪 Evaluation Metrics

- MAE (Mean Absolute Error) for age prediction  
- Accuracy for gender classification  
- Precision, Recall, and F1 Score  

---

## 📌 Highlights

- Low Age MAE (~5.8 years)  
- High Gender Accuracy (>94%)  
- Effective use of ResNet50 transfer learning  
- Multi-task learning approach  
- Clean and modular code structure  

---

## 🧠 Insights

- Deep residual networks extract strong facial features  
- Transfer learning significantly improves performance  
- Gender classification is easier than age estimation  
- Multi-task learning improves efficiency  

---

## 🧪 Future Work

- Add race classification  
- Real-time webcam inference  
- Faster architectures (MobileNet, EfficientNet)  
- Model deployment (web or mobile)  
- Improve age prediction using ordinal regression  

---

## 🙌 Acknowledgements

- UTKFace Dataset  
- PyTorch  
- torchvision (ResNet50 pretrained models)  

---

## 📬 Author

Maryam Pirzadeh  
Deep Learning & Computer Vision Enthusiast  

---

⭐ If you find this project useful, consider giving it a star!
