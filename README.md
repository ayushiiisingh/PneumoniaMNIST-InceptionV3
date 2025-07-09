# Pneumonia Detection with InceptionV3

## Dataset
- PneumoniaMNIST (Grayscale 28x28 X-rays)

## Model
- InceptionV3 (Transfer Learning)
- Top 50 layers fine-tuned
- GlobalAveragePooling + Dropout + Dense layers

## Training Command
```bash
python train_inceptionv3.py
