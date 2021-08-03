# Fake-News-Classifier

#### Goal is to develop a machine learning program to identify when an article might be a fake news.

#### Source: Kaggle- https://www.kaggle.com/c/fake-news/data

A full training dataset with the following attributes:

id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
1: unreliable
0: reliable

#### Steps taken:
##### Step 1:
Preprocessed the text data of the titles, removed stop words, used Porter Stemmer for Stemming of words. Then implemented Bag Of Words using CountVectorizer. 
##### Step 2:
Created Train_test_split and built Multinomial Naive Bayes Classifier. Received an accuracy of 90.2%. Created a Confusion Matrix to identify false positives and false negatives
##### Step 3: 
In order to improve the classification further, used PassiveAggressiveClassifier to build a second classifier. As expected, the classification improved by 2%. Accuracy achieved is 91.8%
##### Step 4: 
Performed Hypertuning on alpha for Multinomail NB. 
##### Step 5: 
Identfied some of the feature names that have highest and lowest coefficients in real and fake segments. 

Improvements:
Use lemmatization instead of stemming, TfIdfVector/Word2Vector instead of CountVectorizer to see how it improves efficiency.
Perform analysis on text instead of titles.

Performed the same steps on the given data but this time, I used TfIdfVectorizer instead of CV and performed the analysis on text field instead of the title field. Accuracy of the model went up to 95.1 when I implemented it using PassiveAggressiveClassifier. Did not use lemmatization as stemming itself took a lot of time to be implemented on texts.

