# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:08:22 2019

@author: Mehadi Hassan
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report


# train Data
trainData = pd.read_csv("IMDB Dataset.csv")
# test Data
testData = pd.read_csv("test.csv")



# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['review'])
test_vectors = vectorizer.transform(testData['review'])


# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['sentiment'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results

print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(testData['sentiment'], prediction_linear, output_dict=True)
print('positive: ', report['positive'])
print('negative: ', report['negative'])



#f1-score = 2 * ((precision * recall)/(precision + recall))


review = """I absolutley liked the product."""
review_vector = vectorizer.transform([review]) # vectorizing
print(classifier_linear.predict(review_vector))