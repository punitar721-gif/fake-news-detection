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
