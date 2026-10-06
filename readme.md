# CAPTCHA Solver

A small computer-vision experiment for training a convolutional neural network to classify CAPTCHA characters from grayscale image samples.

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn

## How It Works

The training script loads PNG samples from `data/train/`, derives labels from file names, converts images to 50×50 grayscale tensors, splits the dataset into training and validation sets, trains a CNN, and saves the trained model and label encoder.

## Model Architecture

```text
Input 50x50x1
  ↓
Conv2D(32) + MaxPooling
  ↓
Conv2D(64) + MaxPooling
  ↓
Flatten
  ↓
Dense(128) + Dropout
  ↓
Softmax output
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

Expected local structure:

```text
data/
└── train/
    ├── A_001.png
    ├── B_001.png
    └── ...

model/
scripts/
└── train_model.py
```

Run training from the `scripts` directory:

```bash
python train_model.py
```

Generated artifacts:

- `model/captcha_model.h5`
- `model/label_encoder.pkl`

## Notes

This repository is an experimental learning project for image classification and model training.
