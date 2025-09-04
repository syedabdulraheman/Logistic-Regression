import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\abdul raheman\OneDrive\python\naresh_it Notes\logit classification.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y , test_size=0.20,random_state=0)

#feature selection technique (standardization / Normalization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =  sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from sklearn.preprocessing import Normalizer
#nr = Normalizer()
#X_train =  nr.fit_transform(X_train)
#X_test = nr.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train , y_train)

y_pred = classifier.predict(X_test)

#model accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#model accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

#clasification maodel
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

#training score
bias = classifier.score(X_train, y_train)
print(bias)

#testing score
variance = classifier.score(X_test, y_test)
print(variance)

#model testing


