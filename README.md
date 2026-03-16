# LSTM Next Word Prediction using RNN

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

**A production-ready Next Word Prediction system using LSTM Neural Networks**

[📖 Overview](#overview) • [⚙️ Workflow](#workflow) • [🏗️ Architecture](#architecture) • [🚀 Getting Started](#getting-started) • [📊 Results](#results) • [🤝 Contributing](#contributing)

</div>

---

## 📖 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Workflow](#system-workflow)
- [Model Architecture](#model-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Prediction](#prediction)
- [API Usage](#api-usage)
- [Results & Performance](#results--performance)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## 📖 Overview

This project implements a state-of-the-art **Next Word Prediction** system using **Long Short-Term Memory (LSTM)** networks. It's designed to predict the most probable next word given a sequence of words, mimicking human language understanding and generation capabilities.

### 🎯 Real-World Applications
- **Search Engines**: Google Search autocomplete
- **Messaging Apps**: WhatsApp, iMessage predictive text
- **Code Editors**: VS Code, PyCharm code completion
- **Writing Assistants**: Grammarly, Microsoft Editor
- **Chatbots**: Context-aware response generation
- **Language Translation**: Sequence-to-sequence models

---

## ✨ Features

| Category | Features |
|----------|----------|
| **Core Functionality** | • Next word prediction<br>• Top-k word suggestions<br>• Temperature-based sampling<br>• Beam search decoding |
| **Model Capabilities** | • Multi-layer LSTM architecture<br>• Dropout regularization<br>• Early stopping<br>• Learning rate scheduling<br>• Model checkpointing |
| **Data Processing** | • Automatic text cleaning<br>• Custom tokenization<br>• Sequence generation<br>• Padding & truncation<br>• Vocabulary management |
| **User Interface** | • Interactive prediction mode<br>• Batch prediction<br>• Command-line interface<br>• Python API<br>• Real-time suggestions |
| **Performance** | • GPU acceleration support<br>• Optimized inference<br>• Model compression<br>• Caching mechanism |

---

## ⚙️ System Workflow
┌────────────────────────────┐
│ COMPLETE SYSTEM WORKFLOW │
└────────────────────────────┘
          |
          ↓
┌──────────────┐
│ INPUT TEXT │
│ "The quick" │
└──────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA PREPROCESSING │
├─────────────────────────────────────────────────────────────────┤
│ │
│ Raw Text → Clean Text → Tokenize → Create Sequences → Pad │
│ │
│ Steps: │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 1. Lowercase conversion │ │
│ │ 2. Remove special characters & punctuation │ │
│ │ 3. Remove extra whitespaces │ │
│ │ 4. Split into words │ │
│ │ 5. Create word-to-index mapping │ │
│ │ 6. Generate n-gram sequences │ │
│ │ 7. Pad sequences to equal length │ │
│ └─────────────────────────────────────────────────────────┘ │
│ │
│ Example: │
│ Input: "The quick brown fox jumps" │
│ Sequences: ["The quick", "The quick brown", │
│ "The quick brown fox", ...] │
│ │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: MODEL ARCHITECTURE │
├─────────────────────────────────────────────────────────────────┤
│ │
│ Input Sequence (length=10) │
│ ↓ │
│ ┌─────────────────┐ │
│ │ Embedding Layer │ (100 dimensions) │
│ │ vocab_size→100 │ │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ LSTM Layer 1 │ (128 units, return_sequences=True) │
│ │ │ Activation: tanh │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ Dropout Layer │ (rate=0.2) │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ LSTM Layer 2 │ (128 units) │
│ │ │ Activation: tanh │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ Dropout Layer │ (rate=0.2) │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ Dense Layer │ (512 units, ReLU) │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ Dropout Layer │ (rate=0.2) │
│ └─────────────────┘ │
│ ↓ │
│ ┌─────────────────┐ │
│ │ Output Layer │ (vocab_size, Softmax) │
│ └─────────────────┘ │
│ ↓ │
│ Probability Distribution over Vocabulary │
│ │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: TRAINING PROCESS │
├─────────────────────────────────────────────────────────────────┤
│ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TRAINING LOOP │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ │ │
│ │ For each epoch: │ │
│ │ ├── Forward pass through network │ │
│ │ ├── Calculate loss (categorical_crossentropy) │ │
│ │ ├── Backward pass (backpropagation) │ │
│ │ ├── Update weights (Adam optimizer) │ │
│ │ ├── Calculate accuracy │ │
│ │ └── Validate on validation set │ │
│ │ │ │
│ │ Monitoring: │ │
│ │ ├── Early stopping if no improvement │ │
│ │ ├── Model checkpoint for best weights │ │
│ │ └── Learning rate reduction on plateau │ │
│ │ │ │
│ └─────────────────────────────────────────────────────────┘ │
│ │
│ Training Configuration: │
│ • Epochs: 100 (with early stopping) │
│ • Batch Size: 32 │
│ • Optimizer: Adam (lr=0.001) │
│ • Loss: Categorical Crossentropy │
│ • Metrics: Accuracy │
│ • Validation Split: 20% │
│ │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: PREDICTION PIPELINE │
├─────────────────────────────────────────────────────────────────┤
│ │
│ User Input: "The quick brown" │
│ ↓ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ PREDICTION PROCESS │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ │ │
│ │ 1. Tokenize input text │ │
│ │ ["the", "quick", "brown"] → [5, 23, 45] │ │
│ │ │ │
│ │ 2. Pad sequence to required length │ │
│ │ [5, 23, 45] → [0, 0, 0, 5, 23, 45] (if length=6) │ │
│ │ │ │
│ │ 3. Pass through model │ │
│ │ Model predicts probability distribution │ │
│ │ │ │
│ │ 4. Get top-k predictions │ │
│ │ Word 1: "fox" (0.75) │ │
│ │ Word 2: "dog" (0.15) │ │
│ │ Word 3: "jumps" (0.05) │ │
│ │ │ │
│ │ 5. Apply temperature (optional) │ │
│ │ Adjusts prediction randomness │ │
│ │ │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ↓ │
│ Output: ["fox", "dog", "jumps"] │
│ │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────┐
│ PREDICTION │
│ "fox" │
└─────────────────┘


---

## 🏗️ Model Architecture Details

### Layer-by-Layer Breakdown

```python
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 10, 100)           500,000
_________________________________________________________________
lstm_1 (LSTM)                (None, 10, 128)           117,248
_________________________________________________________________
dropout_1 (Dropout)          (None, 10, 128)           0
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               131,584
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               66,048
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 5000)              2,565,000
=================================================================
Total params: 3,379,880
Trainable params: 3,379,880
Non-trainable params: 0
_________________________________________________________________

┌─────────────────────────────────────────────────────────────┐
│                      TECHNOLOGY STACK                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Frontend   │  │   Backend    │  │    ML/DL     │       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤       │
│  │ • Terminal   │  │ • Python 3.8 │  │ • TensorFlow │       │
│  │ • CLI        │  │ • Flask (opt)│  │ • Keras      │       │
│  │ • API        │  │ • FastAPI    │  │ • NumPy      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │    Tools     │  │  Deployment  │  │   Testing    │       │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤       │
│  │ • Git        │  │ • GitHub     │  │ • unittest   │       │
│  │ • VS Code    │  │ • Local      │  │ • pytest     │       │
│  │ • Jupyter    │  │ • Docker     │  │ • coverage   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
