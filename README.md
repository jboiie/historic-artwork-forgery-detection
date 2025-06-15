
# Historic Artwork Forgery Detection

A machine learning project to classify historic artworks and flag potential forgeries using deep learning, CNNs, and image metadata analysis.

---

##  Project Overview

Historic artworks are essential to cultural heritage, but authenticating and cataloging them at scale remains challenging.  
This project applies artificial intelligence—using convolutional neural networks (CNNs)—to automate the classification and metadata extraction of artwork images.  
We also explore GAN-based approaches to detect suspicious or potentially manipulated artwork, supporting researchers, museums, and collectors.

---

## 📁 Dataset

- **Source:** [Kaggle Historic Art Dataset](https://www.kaggle.com/datasets/ansonnnnn/historic-art)
- **Size:** 45,000+ artworks with metadata and image files.
- **Core attributes:** artist, title, period, nationality, base, image_path.

> **Note:** Due to storage limits, images are **not included** in this repository—please download them from Kaggle.

---

## 📊 Exploratory Data Analysis (EDA)

Our EDA revealed:
- A handful of artists and periods account for most records (class imbalance).
- Italian, French, Dutch, and Flemish artworks dominate the collection.
- Base locations like Florence, Paris, and Venice are especially well represented.
- Many artworks share generic titles (e.g., “exterior view”, “self-portrait”), which are not always useful for unique identification.

Key plots included distributions of top artists, periods, bases, and nationalities.

---

## ⚙️ Methodology

- **Data Cleaning:** Merged metadata, mapped image paths, removed duplicates and missing entries.
- **Model:** Used a ResNet-50 based CNN for multitask prediction (artist, period, nationality, base).
- **Training:** Split the dataset for supervised learning, evaluated with accuracy and confusion matrices.
- **Forgery Detection:** Used GANs to generate synthetic images and trained classifiers to separate real and generated artworks.

---

## 💾 Model & Data Files

- Model files (`.pth`, `.pkl`, etc.) are stored in `models/`, but large files are not tracked by git.
- If you need large models, contact the authors or download as directed in the README.

---

## 🗂️ Folder Structure

```

historic-artwork-forgery-detection/
│
├── data/                 
├── images/               
├── models/               
├── prediction/           
├── utils/                
├── eda/                  
├── GAN/                  
├── requirements.txt      
├── .gitignore            
└── README.md             

````

---

## ❌ .gitignore Notes

We ignore:
- Images and large data files (`*.pth`, `*.joblib`, etc.)
- Presentations and office docs (`*.pptx`, `*.docx`, etc.)
- Python/OS cache files (`__pycache__`, `.DS_Store`)
- VSCode/system files (`.vscode/`, `Thumbs.db`)

---


## 🏆 Results

* Multitask ResNet-50 model achieves high accuracy for common artists and periods.
* GAN-based forgery detection can flag many synthetic artworks, but is still experimental.
* Limitations include class imbalance and visual similarity across styles.

---

## ⚠️ Limitations & Future Work

* The dataset is imbalanced (some artists/periods overrepresented).
* Forgery detection can be improved with more diverse training data.
* Future work: integrate more metadata fields and advanced anomaly detection.

---

## 👥 Authors

Kaustubh Gothankar (C)
Jai
Anirudh Sahijwani 
Prakriti Chaddha 
Avantika Nair 
Jyotirmoyee Paul 
---

## 📄 References

* [Kaggle Historic Art Dataset](https://www.kaggle.com/datasets/ansonnnnn/historic-art)
* [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), [Scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/)
* Course/project materials, mentors, and teaching assistants

---

## 💬 Contact

For questions or collaboration, open an issue or email \ jai19kharb@gmail.com


