# Facial Emotion Recognition using CNN
| Name                  | Roll Number |
|-----------------------|-------------|
| Ashish Batham         | 202210101150087 |
| Yash Mishra         | 202210101150094 |
| Shashank Kumar        | 202210101150102 |

![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)


A deep learning-based solution for **real-time facial emotion classification** that detects and classifies emotions (happy, sad, angry, surprised, neutral, etc.) from face images. This system can be integrated into e-learning platforms, human-computer interaction systems, and behavioural analytics tools to create emotionally aware applications.

## Problem Statement

Human emotions expressed through facial expressions play a critical role in communication, behaviour, and user experience. Yet, many systems still ignore emotional context or treat facial images in a simplistic way.

**Goal:** Build a robust pipeline that:
- Accepts input face images
- Detects and classifies the underlying emotion
- Enables downstream applications to act on emotional feedback

By capturing and interpreting facial emotion automatically, this system powers **adaptive interfaces**, **user affect monitoring**, and **enriched interactive experiences**.

## python libraries that are used:
  * numpy
  * matplotlib
  * seaborn
  * OpenCV
  * Pillow
  * tensorflow
  * scikit-learn

## Model Architecture

This project implements a **Convolutional Neural Network (CNN)** trained on a standard facial emotion dataset.

### High-Level Architecture

1. **Input & Pre-processing**
   - Face detection using OpenCV Haar cascade
   - Crop and resize face ROI to fixed size
   - Convert to grayscale and normalize pixel values
   - Data augmentation: rotation, horizontal flip, width/height shift

2. **Convolutional Base**
   - Multiple `Conv2D` layers with `3×3` kernels
   - ReLU activation + Batch Normalization
   - `MaxPooling2D` (2×2) for downsampling
   - Progressive increase in depth

3. **Fully-Connected Head**
   - Flatten
   - Dense layers with Dropout to prevent overfitting
   - Final Dense layer with **Softmax** activation for multi-class probabilities

4. **Training & Optimisation**
   - **Loss**: Categorical Cross-Entropy
   - **Optimizer**: Adam (with optional learning rate scheduler)
   - Callbacks: Early Stopping, ModelCheckpoint
   - Evaluation metrics: Accuracy, Confusion Matrix

## Dataset

-  [Facial Dataset](https://www.kaggle.com/datasets/chiragsoni/ferdata) with train and test split (48×48 grayscale images, 7 classes)
- Classes: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

## Usage

1. **Install the required libraries**:

   ```bash
   pip install -r requirements.txt
## Future Enhancements

- Increase dataset variety (lighting, ethnicity, occlusion) to improve generalisation.

- Explore transfer-learning with pretrained backbones (e.g., MobileNet, EfficientNet) for improved accuracy and faster convergence.

- Deploy as a web or mobile service for real-time emotion-aware applications.
