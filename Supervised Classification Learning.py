
"""""""""""

Supervised Learning: Classification

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

Data_iris = iris.data

Data_iris = pd.DataFrame(Data_iris, columns= iris.feature_names)

Data_iris['label'] = iris.target


plt.scatter(Data_iris.iloc[:, 2], Data_iris.iloc[:, 3], c=iris.target)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.show()

x = Data_iris.iloc[:, 0:4]
y = Data_iris.iloc[:, 4]

"""""""""
K-NN Concept

"""""""""

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=1)

KNN.fit(x, y)

x_N = np.array([[5.5,4.7,4.9,0.9]])

KNN.predict(x_N)

x_N2 = np.array([[7.5,5.7,3.9,5.9]])

KNN.predict(x_N2)


"""""""""
Model Testting

"""""""""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8,
                                                    random_state = 88, shuffle=True, 
                                                    stratify = y)

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=6, metric='minkowski', p=1)

KNN.fit(X_train, y_train)

predicted_types = KNN.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predicted_types)

"""""""""
Decision Tree

"""""""""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Dt = DecisionTreeClassifier()

Dt.fit(X_train, y_train)

predicted_types_dt = Dt.predict(X_test)

accuracy_score(y_test, predicted_types_dt)

"""""""""
Cross Validation

"""""""""

from sklearn.model_selection import cross_val_score

Scores_Dt = cross_val_score(Dt, x, y, cv = 10)


"""""""""
Naive Bayes

"""""""""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

NB = GaussianNB()

NB.fit(X_train, y_train)

predicted_types_NB = NB.predict(X_test)

accuracy_score(y_test, predicted_types_NB)

Scores_NB = cross_val_score(NB, x, y, cv = 10)


"""""""""
Logistic Regression

"""""""""

from sklearn.datasets import load_breast_cancer

Data_C = load_breast_cancer()

x = Data_C.data
y = Data_C.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, train_size=0.7,
                                                    random_state = 88)

from sklearn.linear_model import LogisticRegression

Lr = LogisticRegression()

Lr.fit(X_train, y_train)

Predicted_classes_Lr = Lr.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, Predicted_classes_Lr)

from sklearn.model_selection import cross_val_score

Scores_Lr = cross_val_score(Lr, x, y, cv = 10)


"""""""""
Evaluation Metrics

"""""""""

from sklearn.metrics import confusion_matrix, classification_report

Conf_mat = confusion_matrix(y_test, Predicted_classes_Lr)

Class_report = classification_report(y_test, Predicted_classes_Lr)

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

y_prob = Lr.predict_proba(X_test)

y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)

plt.plot(FPR, TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_prob)













