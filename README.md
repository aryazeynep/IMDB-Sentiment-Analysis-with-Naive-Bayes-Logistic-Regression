# IMDB-Sentiment-Analysis-with-Naive-Bayes-Logistic-Regression

## Description
This project explores and practices text classification techniques on the IMDB Movie Review Dataset.
We perform binary sentiment classification (positive/negative) using two models: Naive Bayes and Logistic Regression.
The project focuses on preprocessing text data, building custom models, and comparing their performances.

## Project Details
Dataset: IMDB Movie Review Dataset (from Huggingface datasets library)

**Data**
- 25,000 samples for training
- 25,000 samples for testing
- Each sample contains text and label (0 = negative, 1 = positive)

**Workflow**

Preprocessing
- Remove punctuation and numbers
- Convert text to lowercase
- Remove English stop words
- Clean up extra spaces

Naive Bayes Classifier
- Implemented from scratch
- Includes custom smoothing and log-probability calculations

Logistic Regression
- Created custom feature vectors (Top 10,000 words based on bias score)
- Trained Logistic Regression models with max_iter from 1 to 25
- Visualized accuracy trends over iterations

Comparison and Evaluation
- Compared models based on training and test accuracy
- Commented analysis explaining model choice based on results


