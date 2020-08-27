import random
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
fname = "EEG Eye State.txt"
with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 
content = [x.split(",") for x in content]
content = np.array(content, dtype = 'float32')
random.shuffle(content)
score_p = []
                   
x = content[:, :-1]
y = np.array(content[:, -1], dtype = 'int32')
print(x[0])
print(y[0])
X_columns = ['mean', 'standard deviation', 'kurt', 'skewness']
Y_columns = ['label']
X = pd.DataFrame(columns = X_columns)
Y = pd.DataFrame(columns = Y_columns)
for i in range(len(x)):
  X.loc[i] = np.array([np.mean(x[i]), np.std(x[i]), stats.kurtosis(x[i]), stats.skew(x[i])])
  Y.loc[i] = y[i]
  
print(X.head(n=20))

print(Y.head())



# Split the 'features' and 'income' data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y,test_size = 0.2,random_state = 0)


clf = SVC()
clf.fit(X_train1, y_train1)
predicted = clf.predict(X_test1)

print("Accuracy = {}\nPrecision = {}\nRecall = {}\nF1 Score = {}".format(metrics.accuracy_score(y_test1, predicted), metrics.precision_score(y_test1, predicted),metrics.recall_score(y_test1, predicted),metrics.f1_score(y_test1, predicted)))

score_p.append([metrics.accuracy_score(y_test1, predicted), metrics.precision_score(y_test1, predicted),metrics.recall_score(y_test1, predicted),metrics.f1_score(y_test1, predicted)]) 

print(confusion_matrix(y_test1, predicted))


neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train1, y_train1) 
predicted = neigh.predict(X_test1)
print("Accuracy = {}\nPrecision = {}\nRecall = {}\nF1 Score = {}".format(metrics.accuracy_score(y_test1, predicted), metrics.precision_score(y_test1, predicted),metrics.recall_score(y_test1, predicted),metrics.f1_score(y_test1, predicted)))

score_p.append([metrics.accuracy_score(y_test1, predicted), metrics.precision_score(y_test1, predicted),metrics.recall_score(y_test1, predicted),metrics.f1_score(y_test1, predicted)]) 

print(confusion_matrix(y_test1, predicted))


