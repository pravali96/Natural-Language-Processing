# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 13:23:24 2021

@author: prava
"""
import pandas as pd
messages=pd.read_csv('C:/Users/prava/Downloads/SMSSpamCollection.csv',
                     sep='\t', names=["label","message"])
messages.head()

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

# Preprocessing the texts
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split() 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
   
corpus

# Creating bag of words using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500) # There are 6296 unique words, selected 2500 words
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values 
y

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)


# Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
score=accuracy_score(y_test,y_pred)
print(score)
report=classification_report(y_test,y_pred)
print(report)
