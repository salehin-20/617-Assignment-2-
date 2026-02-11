# Installation Guide (WSL)

This document explains how to install all required libraries and run the project
using **WSL (Windows Subsystem for Linux)**.

---

## 1. Navigate to the Project Directory

```bash
cd ~/617-Assignment-2-
```

---

## 2. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

---

## 3. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, your prompt should show:

```
(.venv)
```

---

## 4. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 5. Install Required Python Libraries

```bash
pip install torch torchvision torchaudio
pip install matplotlib scikit-learn pillow
pip install notebook ipykernel
```

---

## 6. Register the Virtual Environment as a Jupyter Kernel

```bash
python -m ipykernel install --user --name assignment2 --display-name "Python (Assignment 2)"
```

---

## 7. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then:
- Open `notebooks/results.ipynb`
- Select **Kernel → Change Kernel → Python (Assignment 2)**
- Click **Kernel → Restart & Run All**

---

## 8. Verify Installation (Optional)

```bash
python -c "import torch; print(torch.__version__)"
python -c "import matplotlib, sklearn; print('Libraries loaded successfully')"
```

---

## Notes

- The dataset is intentionally excluded from the repository.
- The project will automatically use GPU if available.
- All commands are run from the repository root.

---

End of installation guide.
