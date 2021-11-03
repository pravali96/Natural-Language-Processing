# Stock-Sentiment-Analysis-using-News-Headlines
Data Collection:
I imported this data set from world news and stock price shifts data available on Kaggle. The world news is sourced from Yahoo Finance. 
Dataset contains 25 tops news headlines from 2000 to 2016. It consists of 27 columns and 4101 rows - 25 columns for top 25 news headlines, date and label of the class. Record belongs to class 1 if the stock index went up else 0.

Text-Processing:
I first divided the data set into train and test by setting a date as a threshold. Then using regular expressions, I removed special characters and punctuations from the headlines and replace them with a blank space. Then I replaced capital letters with lower case letters so that bag of words doesn't count them twice. I consolidated all 25 headlines into 1 single chunk under headlines and applied count vectorizer to convert these headlines into a vector of features. I fit-transformed the headlines using CountVectorizer. 

Model Building:
I used Multinomial Naive bayes and Random Forests to build the model using processed text.

Evaluation:
I evaluated the models based on confusion matrix, classfication report, precision and accuracy score. I achieved better accuracy using Naive Bayes compared to Random Forests.
