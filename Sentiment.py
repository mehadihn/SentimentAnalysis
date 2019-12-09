# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 09:49:44 2019
@author: Mehadi Hassan
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report


# train Data
trainData = pd.read_csv("train1.csv")
# test Data
testData = pd.read_csv("test1.csv")



# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(trainData['review'])
test_vectors = vectorizer.transform(testData['review'])

print("Started Trainnig Dataset")
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
print()

print("Training time:", time_linear_train)
print("Prediction time:", time_linear_predict)

print()
print("For Positive Data:")
report = classification_report(testData['sentiment'], prediction_linear, output_dict=True)
print("Precision: ",report['positive']['precision'])
print("Recall: ",report['positive']['recall'])
print("f1-score: ",report['positive']['f1-score'])

print()
print("For Negative Data:")
report = classification_report(testData['sentiment'], prediction_linear, output_dict=True)
print("Precision: ",report['negative']['precision'])
print("Recall: ",report['negative']['recall'])
print("f1-score: ",report['negative']['f1-score'])

#print('positive: ', report['positive'])
#print('negative: ', report['negative'])


import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 3
means_frank = (report['positive']['precision'], report['positive']['recall'], report['positive']['f1-score'])
means_guido = (report['negative']['precision'], report['negative']['recall'], report['negative']['f1-score'])

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, means_frank, bar_width,
alpha=opacity,
color='b',
label='Positive')

rects2 = plt.bar(index + bar_width, means_guido, bar_width,
alpha=opacity,
color='g',
label='Negative')

plt.xlabel('Attribute')
plt.ylabel('Scores')
plt.title('Scores by Attributes')
plt.xticks(index + bar_width, ('Precision', 'Recall', 'f1-score'))
plt.legend()

plt.tight_layout()
plt.show()


from tkinter import *
root = Tk()
root.title("Sentiment Analysis")
root.geometry("640x640+0+0")

heading = Label(root, text="Sentiment Analysis App", font=('Liberation Serif',30,"bold"),fg = "steelblue").pack()
label1 = Label(root, text="Enter Your Text:",font=('Liberation Serif',20,"bold"),fg = "black").place(relx=0.5, rely=0.2, anchor=CENTER)
#label1 = Label(root, text="Enter Your Text:",font=('Liberation Serif',20,"bold"),fg = "black").place(x=10, y=200)
name = StringVar()
entry_box = Entry(root, font=("Calibri",20), textvariable = name , width=30).place(relx=0.5, rely=0.35, anchor=CENTER)



def senti():
    print(str(name.get()))
    review = str(name.get())
    review_vector = vectorizer.transform([review])
    print(classifier_linear.predict(review_vector))
    prediction_linear = classifier_linear.predict(review_vector)
    x = ""
    if prediction_linear[0] == 'negative':
        
        x = "Given Text is Negative"

        label2 = Label(root, text=x,font=('Liberation Serif',21,"bold"),fg = "black").place(relx=0.5, rely=0.6, anchor=CENTER)

    else:
        
        x = "Given Text is Positive"
        label2 = Label(root, text=x,font=('Liberation Serif',21,"bold"),fg = "black").place(relx=0.5, rely=0.6, anchor=CENTER)





submit = Button(root, text = "Submit",font=('Liberation Serif',10,"bold"), width=15, height = 2, command = senti).place(relx=0.5, rely=0.45, anchor=CENTER)
root.mainloop()

