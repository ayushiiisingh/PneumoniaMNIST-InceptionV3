# Pneumonia Detection using InceptionV3

## Objective

The goal of this project is to fine-tune a pre-trained InceptionV3 model to classify chest X-ray images as either pneumonia or normal. The model is trained and evaluated using the PneumoniaMNIST dataset.

## Dataset

- **Name**: PneumoniaMNIST (part of the MedMNIST collection)
- **Format**: `.npz` file containing pre-split train, validation, and test sets
- **Preprocessing**:
  - Grayscale images are converted to RGB
  - Resized from 28x28 to 75x75 to match InceptionV3 input requirements

## Task Description

### Model

- Pre-trained InceptionV3 (ImageNet weights) used as the base model
- Custom classification head added:
  - Global Average Pooling layer
  - Dropout (rate = 0.5)
  - Dense layer with ReLU activation
  - Output layer with sigmoid activation for binary classification

### Evaluation Strategy

1. **Evaluation Metrics**:
   - **Accuracy**: Measures overall correctness
   - **F1 Score**: Balances precision and recall, useful for imbalanced data
   - **ROC-AUC**: Evaluates how well the model separates the two classes

2. **Class Imbalance Handling**:
   - Checked the distribution of pneumonia vs. normal cases
   - Used `class_weight` from `sklearn` to assign higher weight to the minority class
   - Applied image augmentation techniques during training to improve generalization

3. **Overfitting Prevention**:
   - Dropout layer used in the model
   - Image augmentation included rotation, shifting, zooming, and horizontal flipping

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 89%   |
| F1 Score  | 0.915 |
| ROC-AUC   | 0.875 |

Visualizations such as the confusion matrix, ROC curve, and metric bar plots are generated as part of the output.

## Requirements

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```
