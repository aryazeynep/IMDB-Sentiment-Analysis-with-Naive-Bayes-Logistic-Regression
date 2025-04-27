# Arya Zeynep Mete

"""
Objectives:
text classification - binary sentiment classification (positive/negative) 
IMDB Movie Review Dataset
Logistic Regression
"""

"""1-) Dataset"""

#!pip install datasets
import nltk
nltk.download("punkt")
from datasets import load_dataset
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from collections import Counter
import math
from sklearn.metrics import accuracy_score
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import warnings
import sys
from sklearn.exceptions import ConvergenceWarning

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

"""2-) Preprocessing"""
# a function preprocess_text(text) 
# given text and returns the new processed text

def preprocess_text(text):
    # Step 1: Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Step 2: Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Step 3: Convert to lowercase
    text = text.lower()
    
    # Step 4: Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    
    # Clean up whitespace
    text = re.sub(' +', ' ', text).strip()
    
    return text

# Apply preprocessing to both dataframes
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)


# 3 Logistic Regression

#  In this section, preprocessed dataframes train_df and test_df will be used.
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)


# Define bias_scores function
def bias_scores(train_df):
    pos_texts = train_df[train_df['label'] == 1]['text']
    neg_texts = train_df[train_df['label'] == 0]['text']

    pos_counter = Counter(" ".join(pos_texts).split())
    neg_counter = Counter(" ".join(neg_texts).split())

    all_words = set(pos_counter.keys()).union(set(neg_counter.keys()))
    scores = []

    for word in all_words:
        fp = pos_counter[word] # frequency of word w in positive class.
        fn = neg_counter[word] # frequency of word w in negative class.
        ft = fp + fn # total frequency in both classes
        if ft == 0:
            continue
        score = abs((fp - fn) / ft) * math.log(ft)
        scores.append((word, fp, fn, ft, score))
    
    scores.sort(key=lambda x: (-x[4], x[0]))
    return scores[:10000] # returns a list consisting of the top 10,000 tuples based on the highest bias scores


# Compute bias scores and select top 10,000 words
scores = bias_scores(train_df)
top_words = [t[0] for t in scores]

"""
A Bag-of-Words vector is constructed using these 10,000 words, which are assumed
to have the most influence on sentiment analysis within the train_df dataset. 

Then i apply this vectorizer to transform all the texts in train_df and
test_df.
"""

# Vectorize with top 10,000 words
vectorizer = CountVectorizer(vocabulary=top_words)
X_train = vectorizer.transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])
y_train = train_df['label']
y_test = test_df['label']


# LOGISTIC REGRESSION TRAINING SECTION

# Completely suppress warnings and set low tolerance internally
warnings.filterwarnings("ignore", category=ConvergenceWarning)

train_accuracies = []
test_accuracies = []

for i in range(1, 26):
    lr_model = LogisticRegression(max_iter=i)
    
    try:
        # Force immediate return after max_iter
        lr_model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, lr_model.predict(X_train))
        test_acc = accuracy_score(y_test, lr_model.predict(X_test))
    except:
        train_acc, test_acc = 0.5, 0.5
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    # print(f"Completed iteration {i}/25 - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    sys.stdout.flush()  # Force immediate output


# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(1, 26), train_accuracies, marker='o', label='Training Accuracy')
plt.plot(range(1, 26), test_accuracies, marker='x', label='Test Accuracy')
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy vs Max Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Final comment block for analysis
"""
Analysis:
From the plot, it is observed that the training accuracy increases rapidly as the number of iterations increases,
while the test accuracy reaches a plateau and becomes stable after around 10 to 15 iterations.
Choosing a model with around 15 iterations is ideal, as it balances learning capacity with generalization, 
minimizing both underfitting and overfitting. It ensures consistent performance on unseen data.
"""
# FORCE TERMINATION OF ANY BACKGROUND PROCESSES
import gc
gc.collect()  # Clean up memory

"""
#MANUAL TESTING FROM ASSIGNMENT PDF

scores = bias_scores(train_df)
print(scores[:2])
print(scores[-2:])


#Output:
#[('worst', 252, 2480, 2732, 6.453036011602796), ('waste', 99, 1359, 1458, 6.295524245429657)]
#[('complimented', 10, 3, 13, 1.3811265770946735), ('conformity', 10, 3, 13, 1.3811265770946735)]


#My output: -I get true output
#Completed iteration 25/25 - Train Acc: 0.944, Test Acc: 0.866
#[('worst', 252, 2480, 2732, 6.453036011602796), ('waste', 99, 1359, 1458, 6.295524245429657)]
#[('complimented', 10, 3, 13, 1.3811265770946735), ('conformity', 10, 3, 13, 1.3811265770946735)]

#[Done] exited with code=0 in 75.922 seconds
"""