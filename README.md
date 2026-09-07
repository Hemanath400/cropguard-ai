🌱 CropGuard AI

**AI-Powered Tomato Disease Detection & Crop Management Assistant**

CropGuard AI is a full-stack agricultural intelligence platform that helps farmers identify tomato plant diseases from leaf images using Deep Learning and receive AI-generated management recommendations.

The platform combines Computer Vision, Machine Learning, and Generative AI to provide fast and accessible crop health insights through a modern web application.

---

## 🚀 Live Demo

**Website:** [cropguard-ai-nine.vercel.app](https://cropguard-ai-nine.vercel.app/)

---

## 📖 Overview

Crop diseases can significantly reduce crop yield and quality if not identified early. **CropGuard AI** aims to assist farmers by providing an intelligent disease detection system that **analyzes tomato leaf images and generates actionable recommendations.**

Users simply upload an image of a tomato leaf, and the system predicts the disease category along with a confidence score. Based on the prediction, AI-generated management advice is provided to help users understand and respond to the issue.

---

## ✨ Features

* 🌿 Tomato leaf disease detection
* 🤖 Deep Learning-powered image classification
* 📊 Confidence score prediction
* 🧠 AI-generated disease management recommendations
* ⚡ FastAPI backend for inference
* 🎨 Modern responsive web interface
* ☁️ Cloud deployment for public access

---

## 🦠 Supported Disease Classes

The current model supports the following 10 classes:

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy

---

## 🏗️ System Architecture

```text
		👨‍🌾 USER
            │
            ▼
     🌐 CropGuard AI 
     	HTML / CSS
            │
            ▼
       ⚡ FastAPI API
            │
            ▼
   🤗 Hugging Face Model Repository
            │
            ▼
 🧠 ResNet50 Model TensorFlow
            │
            ▼
     Disease + Confidence Score
            │
            ▼
      🤖 GenAI Layer
            │
            ▼
            │
            ▼
          👨‍🌾 USER
```

---

## 🛠️ Tech Stack

### Frontend

* Html
* CSS

### Backend

* FastAPI
* Python

### Machine Learning

* TensorFlow
* ResNet50 Transfer Learning
* Keras
* NumPy
* Pillow

### Model Hosting

* Hugging Face Hub

The trained ResNet50 model is hosted on Hugging Face and loaded by the backend during inference.

### Generative AI

* Gemini API

Used to generate disease-specific crop management guidance based on the prediction.

### Deployment

* Vercel

The web application is deployed and accessible through the live CropGuard AI website.

---

## 📊 Current Version (v1)

CropGuard AI v1 focuses on:

* Tomato disease classification
* AI-powered crop management advice
* End-to-end web deployment
* User-friendly prediction workflow

This version serves as the foundation for a larger agricultural intelligence platform.

---

## 🔍 Key Learning

During testing, an important observation was identified:

The model performs strongly on images similar to the training dataset but can struggle with certain real-world unseen images due to differences in:

* Lighting conditions
* Camera quality
* Leaf orientation
* Background variation
* Disease severity levels

This highlights the importance of domain generalization and real-world validation in agricultural AI systems.

Future versions will focus heavily on improving robustness through dataset expansion and field testing.

---

## 🌾 Roadmap

### CropGuard AI v2

* Weather intelligence integration
* Farm profile management
* Crop recommendation system
* Disease risk prediction
* Agricultural knowledge base
* Real-world farm validation
* Context-aware AI advisor

### CropGuard AI v3

* Pest identification
* Nutrient deficiency detection
* Yield prediction
* Farm health dashboard
* Multi-crop support

---

## 🎯 Vision

CropGuard AI aims to evolve from a disease detection application into a comprehensive agricultural intelligence platform that helps farmers:

* Detect crop issues
* Understand risks
* Predict future problems
* Make informed farming decisions

The long-term goal is to provide practical, AI-driven support for real-world agricultural operations.

---

## 👨‍💻 Developer

**Hemanath T**

GitHub: [github.com/Hemanath400](https://github.com/Hemanath400)

LinkedIn: [www.linkedin.com/in/hemnaththangavel](https://www.linkedin.com/in/hemnaththangavel/)

Portfolio: [hemanath-portfolio.vercel.app](https://hemanath-portfolio.vercel.app/)

---

## ⭐ Support

If you find CropGuard AI useful, consider starring the repository and sharing feedback to help improve future versions.
