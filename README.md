# NLP Document Classification Pipeline

This project demonstrates a complete Natural Language Processing (NLP) pipeline for transforming unstructured text into structured features and applying supervised machine learning models.

---

## 🎯 Project Goal
Build a machine learning system that classifies text data (e.g. messages, reviews, emails) into categories such as **Spam vs Not Spam** or **Positive vs Negative**.

---

## ⚙️ Pipeline Overview
- Text preprocessing and normalization  
- Feature extraction using **TF-IDF vectorization**  
- Train/test data split  
- Model training using **Multinomial Naive Bayes**  
- Model evaluation using classification metrics  

---

## 🛠 Technologies Used
- Python 3  
- pandas  
- scikit-learn  

---

## 📁 Project Structure
nlp-document-classification-pipeline/
├── README.md
├── main.py
└── requirements.txt
---
## 🚀 How to Run Locally
```bash
# Clone the repository
git clone https://github.com/Ifeanyi-Ogene/nlp-document-classification-pipeline.git

cd nlp-document-classification-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
