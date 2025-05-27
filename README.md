---
title: Item Classifier App
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.31.0"
app_file: item_classifier.py
pinned: false
---

# Item Classifier App

A **Gradio-powered** **neural network** image classification app that lets users upload an image and receive a predicted label. This app demonstrates the use of two deep learning neural network models trained for visual classification:

- A **MobileNetV2** (pretrained convolutional neural network)
- A **Custom CNN** (convolutional neural network trained from scratch)

## Features

- Train model using images in item_data folder or use your own images!
- After model is trained, upload an image and get an instant item prediction powered by neural networks
- Choose between a lightweight **MobileNetV2** or **custom CNN model**
- Built using **TensorFlow** and **Keras** deep learning frameworks
- Dockerized deployment support
- Integrated CI/CD pipeline with **GitHub Actions**
- Live hosted on **Hugging Face Spaces**

## Models

### 1. MobileNetV2 (Pretrained Transfer Learning Model)

MobileNetV2 is a pretrained convolutional neural network designed for efficient image classification tasks.

- Base model: MobileNetV2 pretrained on ImageNet dataset, used as a fixed feature extractor (weights frozen).
- Input shape: 224x224 RGB images.
- Architecture:
  - MobileNetV2 base without the top classification layers.
  - Global Average Pooling layer to reduce spatial dimensions.
  - Dense layer with 64 units and ReLU activation.
  - Output Dense layer with softmax activation corresponding to 3 classes (`Cat`, `Dog`, `Other`).
- Training: Only the top layers are trained on the custom dataset, benefiting from the rich features learned by MobileNetV2 on a large dataset.
- Advantages: High accuracy and good generalization on unseen images due to transfer learning.

### 2. Custom CNN (Trained from Scratch)

A convolutional neural network built and trained fully from scratch on the custom dataset.

- Input shape: 224x224 RGB images.
- Architecture:
  - Conv2D layer with 32 filters, 3x3 kernel, ReLU activation.
  - MaxPooling layer with pool size 2x2.
  - Conv2D layer with 64 filters, 3x3 kernel, ReLU activation.
  - MaxPooling layer with pool size 2x2.
  - Flatten layer.
  - Dense layer with 64 units and ReLU activation.
  - Output Dense layer with softmax activation for 3 classes.
- Training: Trained fully from scratch on the custom dataset.
- Advantages: Simpler architecture but may have lower accuracy on small datasets.

**Note:** Model files are stored as `.h5` and loaded by the app at runtime.

## Training Details

- Data Augmentation: Random rotations, width/height shifts, zoom, and horizontal flips using `ImageDataGenerator` to improve generalization.
- Dataset: Images resized to 224x224 pixels. Currently trained on a small dataset (12 images each for `Cat` and `Dog`).
- Loss Function: Sparse categorical crossentropy.
- Optimizer: Adam optimizer.
- Epochs: 10 epochs.
- Batch size: 8.

## Findings

- **MobileNetV2** is noticeably more accurate than the custom **CNN** when tested on unseen images. This is primarily because MobileNetV2 is a pretrained model built on a large and diverse dataset (ImageNet), allowing it to extract rich and robust features that generalize well to new data.

- In contrast, the custom CNN is trained from scratch on a very small dataset (only 12 images), limiting its ability to learn complex features and generalize beyond the training samples. Transfer learning with MobileNetV2 leverages learned low- and mid-level features that improve accuracy, especially when training data is limited.

## 📊 Dataset

Found in item_data/

- **Classes:** `Cat` and `Dog`
- **Training Size:** 12 images per class (total 24)
- **Image Size:** Resized to `224 x 224` pixels for model input
- Dataset kept intentionally small for experimentation and demonstration purposes

## 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with Docker:

```
docker-compose up
# To stop:
docker-compose down
```

## Testing

Run the included test suite with:
`pytest tests/test.py`

## CI/CD Pipeline

This repository includes a GitHub Actions workflow:

- Automatically runs tests on every push or pull request to the main branch
- If tests pass, the app is deployed to Hugging Face Spaces

## Live Demo

Access the deployed app here:
🔗 https://huggingface.co/spaces/lawrencecodes/item_classifier_model

## Project Structure

```
├── item_classifier.py      # Core logic for loading and running
├── item_classifier_mobilenetv2.h5 # Trained model: MobileNetV2
├── item_classifier_cnn.h5  # Trained model: CNN
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── tests/
│   └── test.py
├── pytest.ini # pytest config
├── item_data/
│   └── Cat
│   └── Dog
└── .github/workflows/ci-cd.yml
```

## Notes

- .gradio/ is .gitignored to prevent local certificates from being committed
- Hugging Face Spaces run on-demand, so the app may rebuild after periods of inactivity
- Model files can be persisted either by bundling them in the repo or uploading to Hugging Face Model Hub

## Author

Lawrence Lee

## Resources

- Hugging Face Spaces Documentation
- Gradio Documentation

## Additional Notes

MobileNetV2 is a versatile neural network architecture widely used beyond image classification. Due to its lightweight design and efficiency, it is commonly applied in various computer vision tasks such as:

- Face mask detection
- Real-time video analysis
- Medical image classification
- Gesture recognition
