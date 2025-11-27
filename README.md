
# ✨ Uncovering Hidden Bias: Rating–Sentiment Discrepancies in Consumer Reviews
### *A NLP + ML Project by Sanah Sarin & Serena Wong*

## 🚀 Overview
This project digs into one big question:  
**Why do people give a 1-star rating but write like they’re in love with the restaurant… and vice-versa?**

Using **42,000+ Yelp restaurant reviews**, we uncover hidden mismatches between numerical star ratings and the actual sentiment embedded in text.  

With classical ML, deep learning models, transformer-based sentiment analysis, and engineered metadata, we identify how certain cuisines, categories, and restaurant types systematically exhibit rating inconsistencies.

This repository contains all code, data processing scripts, models, analyses, and plots used to reproduce our full pipeline.

---

## 🧠 Key Features
✔️ Multi-model star rating prediction (SVM, Logistic Regression, TextCNN, DeepTextCNN, Bi-LSTM)  
✔️ Aspect-Based Sentiment Analysis (Food, Service, Atmosphere)  
✔️ Category-level mismatch detection & visualization  
✔️ Metadata engineering (parking, ambience, categories, attributes)  
✔️ Clean 42k-review dataset (Yelp Open Dataset)  
✔️ Full Agile development workflow using Jira + sprints  

---

## 🔬 Methodology Snapshot

### **Models Used**
- **TF–IDF + SVM** → ~62% accuracy  
- **TF–IDF + Logistic Regression** → ~64%  
- **TextCNN** → ~64%  
- **DeepTextCNN** (GloVe + multi-kernel CNN) → **66.3%**  
- **Bi-LSTM** (GloVe + metadata) → ~63%  

### **Sentiment Methods**
- Overall sentiment: `distilbert-base-uncased-finetuned-sst-2-english`  
- Aspect sentiment: `yangheng/distilbert-base-uncased-absa`  

### **Train/Test Split**
- **StratifiedGroupKFold** to prevent the same business appearing in train + test  
- Maintains star distribution and avoids leakage  

---

## 🏗️ Agile Workflow & Jira

We followed a structured **Agile methodology** with three 1-week sprints:

### **🌀 Sprint Breakdown**
1. **Sprint 1:** Data cleaning, EDA, feature engineering  
2. **Sprint 2:** Classical ML modelling + baselines  
3. **Sprint 3:** Deep models, mismatches, visualizations  

### **📌 Jira Board Practices**
- Backlog grooming  
- Sprint planning  
- Daily task updates  
- Status columns: *To Do → In Progress → In Review → Done*  
- Velocity tracking  

This ensured consistent progress and a clean, reproducible pipeline.

---

## 📊 Results Summary

Our best model, **DeepTextCNN**, achieved **66.3% accuracy**, outperforming classical baselines.

Key findings from mismatch analysis:
- Certain cuisines (e.g., Korean, Italian) showed **10–20% higher mismatch rates**  
- Mid-range ratings (2★–4★) accounted for **~70% of mismatches**, due to ambiguous language  
- Extreme ratings (1★, 5★) were predicted most accurately  
- Businesses with unclear ambience or mixed attributes exhibited more inconsistent reviews  

---

## 🛠️ Tech Stack
**Languages:** Python  
**Libraries:** PyTorch, TensorFlow/Keras, scikit-learn, pandas, NumPy  
**NLP:** HuggingFace Transformers, NLTK  
**Embeddings:** GloVe (100d)  
**Visualization:** matplotlib, seaborn  
**Tools:** Jira, GitHub, Jupyter, Overleaf  

---

## 🧑‍💻 Authors
**Sanah Sarin** and **Serena Wong**

---

## ⭐ Want to Use or Reference This Work?
Give the repo a ⭐ and feel free to cite or fork it!

