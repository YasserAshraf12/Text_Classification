# Author: Aya Mahmoud - Nour Atef - Yasser Ashraf
# IDs: 20170071 - 20170325 - 20170331
# Version: 1.0
# Description: TFIDF Movie Review
#--------------------------------------------------------------
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
#----------------------------------------------------------------
Data = load_files(r"E:\BOOKS AND DOCUMENTS\Level 04\Second Term\NLP\Assignments\Assignment Two\Assignment_NLP\movie_dt",
                  categories=["pos", "neg"])
X, Y = Data.data, Data.target
#--------------------------------------------------------------
# remove unwanted words or special charcters function
def removeChars(line):

    # Remove special chars
    txtFile = re.sub(r'\W', ' ', str(line))

    # remove "a" indefinite articles
    txtFile = re.sub(r'\s+[a-zA-Z]\s+', ' ', txtFile)

    # Remove "the" word
    txtFile = re.sub(r'\s+the\s+', ' ', txtFile)

    # replace many spaces with single space
    txtFile = re.sub(r'\s+', ' ', txtFile, flags=re.I)

    # Converting to Lowercase
    txtFile = txtFile.lower()

    return txtFile
#--------------------------------------------------------------
# calcualte the accuracy of test data
def calAccuracy(y_pred):
    matchTarget = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(y_pred, y_test)]
    accuracy = (sum(matchTarget) / len(matchTarget) * 100)
    print("Model Accuracy is: ", round(accuracy, 2), '%')
#--------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


Files = []
for line in range(0, len(x_train)):
    txtFile = removeChars(x_train[line])

    Files.append(txtFile)
tfidfconverter = TfidfVectorizer(use_idf=True, stop_words=None)
tfidf = tfidfconverter.fit(Files)
x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)
classifier = SGDClassifier().fit(x_train, y_train)
y_pred = classifier.predict(x_test)
calAccuracy(y_pred)
review = input('Please, Enter Your Review: ')
updated_review = removeChars(review)
updated_review = [updated_review]
pred = classifier.predict(tfidf.transform(updated_review))
if (pred == [0]):
    print("you give a bad review")
else:
    print("you give a good review")
#--------------------------------------------------------------
dfx = pd.DataFrame(x_test, columns=['Column_X'])
dfy = pd.DataFrame(y_test, columns = ['Column_Y'])
positive = dfy[dfy['Column_Y'].isin([1])]
negative = dfy[dfy['Column_Y'].isin([0])]
#plt.scatter(x_positive,y_positive, red) for Positive
#plt.scatter(x_negative,y_negative, blue) for Negative
#plt.title("TFIDF Movie Review")
#plt.xlabel("X")
#plt.ylabel("Y")
#plt.show();
#--------------------------------------------------------------
