# Arya Zeynep Mete

"""
Objectives:
text classification - binary sentiment classification (positive/negative) 
IMDB Movie Review Dataset
Naive Bayes
"""

"""1.1-) Dataset"""
# load_dataset("imdb") function from the datasets library
# split into two pandas dataframes: "train" and "test," each containing 25,000 samples. 
# each has two colums: "text" and "label". label contains 0 (negative) and 1

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

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Convert to pandas DataFrames
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

"""1.2-) Preprocessing"""
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


"""2 Naive Bayes"""

class NaiveBayesClassifier:
    def __init__(self):
        # Properties will be initialized in the fit() method
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.prior_pos = 0
        self.prior_neg = 0
        self.pos_counter = Counter()
        self.neg_counter = Counter()
    
    def fit(self, train_df):
        
        # Separate positive and negative texts
        pos_texts = train_df[train_df['label'] == 1]['text']
        neg_texts = train_df[train_df['label'] == 0]['text']

        # Tokenize all words
        pos_words = ' '.join(pos_texts).split()
        neg_words = ' '.join(neg_texts).split()

        # Count words and vocabulary
        self.total_pos_words = len(pos_words)
        self.total_neg_words = len(neg_words)
        self.pos_counter = Counter(pos_words)
        self.neg_counter = Counter(neg_words)
        all_words = set(pos_words + neg_words)
        self.vocab_size = len(all_words)

        # Prior probabilities
        total_samples = len(train_df)
        self.prior_pos = len(pos_texts) / total_samples
        self.prior_neg = len(neg_texts) / total_samples

    def predict(self, text):
        # Preprocess input text
        processed_text = preprocess_text(text)
        words = processed_text.split()

        # Initialize log probabilities with class priors
        log_prob_pos = math.log(self.prior_pos)
        log_prob_neg = math.log(self.prior_neg)

        for word in words:
            # Calculate smoothed likelihoods
            word_pos_count = self.pos_counter.get(word, 0)
            word_neg_count = self.neg_counter.get(word, 0)

            # Apply Laplace smoothing
            prob_word_given_pos = (word_pos_count + 1) / (self.total_pos_words + self.vocab_size)
            prob_word_given_neg = (word_neg_count + 1) / (self.total_neg_words + self.vocab_size)

            # Add log probabilities
            log_prob_pos += math.log(prob_word_given_pos)
            log_prob_neg += math.log(prob_word_given_neg)

        # Predict based on higher log probability
        y_predicted = 1 if log_prob_pos > log_prob_neg else 0

        return y_predicted, log_prob_pos, log_prob_neg
    
"""
#MANUAL TESTING FROM ASSIGNMENT PDF

nb=NaiveBayesClassifier()
nb.fit(train_df)

print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])


#Output:  -I get true output
#1575152
#1516208
#74002
#0.5
#0.5
#6419
#2642


prediction1=nb.predict(test_df.iloc[0]["text"])
prediction2=nb.predict("This movie will be place at 1st in my favourite movies!")
prediction3=nb.predict("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment.")
print(f"{'Positive' if prediction1[0]==1 else 'Negative'}")
print(prediction1)
print(f"{'Positive' if prediction2[0]==1 else'Negative'}")
print(prediction2)
print(f"{'Positive' if prediction3[0]==1 else'Negative'}")
print(prediction3)


# Output: -I get true output
# Negative
# (0,-1167.5758675517511,-1146.4479616999306)
# Positive
# (1,-36.43364380516184,-37.068841883770205) 
# Negative
# (0,-57.05497089563332,-53.21115758896025)


print(preprocess_text("This movie will be place at 1st in my favourite movies!"))
print(preprocess_text("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment."))


# Output: -I get true output
# movie place st favourite movies
# wait movie end turned halfway complete disappointment


y_true = test_df['label'].values
y_pred = [nb.predict(text)[0] for text in test_df['text']]
# from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")


# Output: -I get true output
# Accuracy: 0.82464
"""

