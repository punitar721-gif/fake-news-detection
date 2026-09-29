# Fake News Detection Using Machine Learning

## 📌 Project Overview

This project is a machine learning system that classifies a news article as **potentially fake or potentially real** based on its text.

The system uses **TF-IDF** to convert news text into numerical features and **Logistic Regression** to perform the classification.

> Note: The model's prediction is based on patterns learned from the dataset. It does not independently verify whether a real-world news article is factually true or false.

## 🎯 Objectives

- Detect potentially fake and potentially real news articles.
- Apply Natural Language Processing (NLP) techniques to news text.
- Train a machine learning classification model.
- Evaluate the model using standard performance metrics.
- Provide predictions for new text entered by the user.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Google Colab
- GitHub

## 📂 Dataset

The project uses the **Fake and Real News Dataset**, containing labeled news articles.
The dataset is available on Kaggle:

[Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

The dataset files are not included in this repository because of their large file size.
The dataset contains two categories:

- Fake News
- Real News

The dataset is used for educational and machine learning experimentation.

## 🔄 Methodology

```text
News Article
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Prediction
     ↓
Potentially Fake / Potentially Real
## 🤖 Machine Learning Model

**Algorithm:** Logistic Regression

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features that can be processed by the machine learning model.

## 📊 Model Evaluation

The model was evaluated using a separate test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 98.46% |
| Precision | 98.24% |
| Recall | 98.52% |
| F1-Score | 98.38% |

These results are based on the test split used in this project.

## 📁 Project Files

- `Fake_News_Detection.ipynb` — Complete machine learning notebook
- `fake_news_detector.py` — Python project file
- `fake_news_model.pkl` — Trained Logistic Regression model
- `tfidf_vectorizer.pkl` — Trained TF-IDF vectorizer
- `data/` — Dataset information

## 🚀 How to Run

1. Open `Fake_News_Detection.ipynb` in Google Colab.
2. Upload the dataset files.
3. Run the notebook cells in order.
4. Train the model.
5. Enter a news article when prompted.
6. View the model's prediction.

## 🔮 Future Enhancements

- Add advanced NLP preprocessing.
- Compare multiple machine learning algorithms.
- Develop a web-based user interface.
- Add a larger and more diverse dataset.
- Improve handling of short news text.
- Add explainable AI features.

## 👩‍💻 Author

**Punitha R R**

This project was developed as a machine learning project for educational purposes.
