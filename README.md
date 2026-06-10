#  Fake News Detector

> A machine learning web app that detects whether a news article is **Real or Fake**, trained on 44,898 real-world articles with 99.7% accuracy.

##  Model Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 99.35% | 99.45% | 99.32% | 99.38% |
| Naive Bayes | 95.59% | 95.26% | 96.36% | 95.81% |
| **Random Forest**  | **99.73%** | **99.77%** | **99.72%** | **99.74%** |

##  Features

-  Paste any news headline or article → instant Real/Fake prediction
-  Confidence score with probability bar chart
-  SHAP explainability — see exactly which words drove the decision
-  Confusion matrices for all 3 models
-  Best model auto-selected (Random Forest)

##  Tech Stack

`Python` `scikit-learn` `Streamlit` `SHAP` `pandas` `numpy` `TF-IDF`

##  Dataset

ISOT Fake News Dataset — 21,417 real + 23,481 fake articles  
Source: [clmentbisaillon on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

##  How to Run Locally

```bash
git clone https://github.com/FatalDestinyXO/Fake-News-Detector
cd Fake-News-Detector
pip install -r requirements.txt
python train.py
streamlit run app.py
```

App opens at `http://localhost:8501`

##  Project Structure

```
fake-news-detector/
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── preprocess.py       # Text cleaning utilities
├── explain.py          # SHAP explainability module
├── models/             # Saved trained models (.pkl)
├── data/               # Dataset (True.csv + Fake.csv)
├── notebooks/          # EDA Jupyter notebook
└── requirements.txt
```