# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:19:08 2021

@author: prava
"""
import pandas as pd
df=pd.read_csv("C:/Users/prava/Downloads/fakenews/train.csv")

X=df.drop('label', axis=1)
X.head()

y=df['label']

df.shape

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, HashingVectorizer

df=df.dropna()

df.shape

messages=df.copy()

messages.reset_index(inplace=True)

# messages['text'][6]

# Preprocessing text using regular expression:
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer
import re
ps=PorterStemmer()
corpus=[]
for i in range(0, len(messages)):
    review= re.sub('[^a-zA-z]', ' ', messages['text'][i])
    review=review.lower()
    review=review.split() # we will have list of words after split
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
# Applying countVectorizer to create a bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,3)) 
# takes top 5000 common occuring words and also words that occurs in combinations of 1-3 words
X= tfidf.fit_transform(corpus).toarray()

X.shape

y=messages['label']

# Creating Train tets split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)

tfidf.get_feature_names()[:20]

tfidf.get_params() #gives info of the vector

count_df=pd.DataFrame(X_train,columns=tfidf.get_feature_names())
count_df.head()
count_df.shape

import matplotlib.pyplot as plt
import numpy as np
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Multinomial NB Algo
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

classifier.fit(X_train, y_train)
pred=classifier.predict(X_test)

# Calculating Performance of MultinomailNB
from sklearn import metrics
score = metrics.accuracy_score(y_test, pred)
print("accuracy: %0.3f" %score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

# Accuracy of 0.902

# Passive Aggressive Classifier Algorithm
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf=PassiveAggressiveClassifier(n_iter_no_change=50)

linear_clf.fit(X_train, y_train)
pred2=linear_clf.predict(X_test)

# Calculating Performance of PassiveAggressiveClassifier
score = metrics.accuracy_score(y_test, pred2)
print("accuracy: %0.3f" %score)
cm = metrics.confusion_matrix(y_test, pred2)
plot_confusion_matrix(cm, classes=['Fake', 'Real'])

# Accuracy 95.1, performed better than MultinomialNB

# Multinomial Classifier with hyperparameter tuning
classifier=MultinomialNB(alpha=0.1)

previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))
    
'''
Alpha: 0.0, Score : 0.9032081662413416
Alpha: 0.1, Score : 0.9026613197229311
Alpha: 0.2, Score : 0.9026613197229311
Alpha: 0.30000000000000004, Score : 0.9030258840685381
Alpha: 0.4, Score : 0.9032081662413416
Alpha: 0.5, Score : 0.9028436018957346
Alpha: 0.6000000000000001, Score : 0.9024790375501276
Alpha: 0.7000000000000001, Score : 0.9022967553773241
Alpha: 0.8, Score : 0.9028436018957346
Alpha: 0.9, Score : 0.9021144732045207
'''
# Identifying most real and fake news:

# Get Features names
feature_names = tfidf.get_feature_names()   
classifier.coef_[0]  

### Most real---highest value represents that it could be real
sorted(zip(classifier.coef_[0], feature_names), reverse=True)[:20]

### Most fake---Lowest value represents that it could be fake
sorted(zip(classifier.coef_[0], feature_names))[:50]
