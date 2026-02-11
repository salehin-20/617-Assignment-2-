# Assignment 2 – Garbage Classification Model (PyTorch)

This repository contains the solution for **Assignment 2: Garbage Classification Model**.  
The task is to build a **multimodal classification system** using **both images and textual information**, implemented in **PyTorch**.

The model predicts the correct disposal category for a garbage item:
- **Black**
- **Blue**
- **Green**
- **Other**

---

## Repository Contents

```
.
├── src/
│   ├── dataset.py      # Multimodal dataset (image + filename text)
│   ├── model.py        # Image–text fusion neural network
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script
│
├── notebooks/
│   └── results.ipynb   # Model predictions, metrics, confusion matrix,
│                       # and incorrect classification visualizations
│
├── checkpoints/
│   └── best.pt         # Best trained model (saved by validation accuracy)
│
├── README.md
└── .gitignore
```

---

## Dataset Description

- The dataset is organized into four folders:
  ```
  Black / Blue / Green / Other
  ```
- Each image contains **one object**.
- The **textual modality** is derived from the image filename  
  (underscores replaced with spaces).
- Supported image formats: `.jpg`, `.jpeg`, `.png`
- The dataset itself is **not included** in this repository, as required.

---

## Model Overview

The model is a **multimodal neural network**:

- **Image branch:** Pretrained ResNet-18
- **Text branch:** Character-level embedding of filename text
- **Fusion:** Concatenation of image and text features
- **Output:** 4-class softmax classifier

---

## Training

To train the model:

```bash
python src/train.py --data_dir dataset --epochs 10 --batch_size 16
```

- The best model (based on validation accuracy) is saved to:
  ```
  checkpoints/best.pt
  ```

---

## Evaluation

To evaluate the trained model:

```bash
python src/evaluate.py --data_dir dataset --ckpt checkpoints/best.pt
```

Evaluation includes:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

---

## Results Notebook

The file:

```
notebooks/results.ipynb
```

contains:
- Classification metrics
- Confusion matrix visualization
- **Figures of incorrect classifications** (image, text, true label, predicted label)

This notebook satisfies the **mandatory deliverable** specified in the assignment.

---

## Results Summary

- Validation accuracy achieved: **~85%**
- Strong performance on **Green** and **Blue** categories
- Some confusion between **Black** and **Other**, addressed in error analysis

---

## Submission Notes

- This repository is the **only submission** for Assignment 2.
- The dataset is intentionally excluded.
- The submission on D2L consists of **the GitHub repository link only**.

---

## Course Information

Assignment 2 – Garbage Classification Model  
Course: **ENSF 617**  
Implementation uses **PyTorch** and follows techniques covered in class.
