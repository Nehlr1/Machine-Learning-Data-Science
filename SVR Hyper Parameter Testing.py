"""""""""""
SVR Hyper Parameter Testing

"""""""""""

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

x = Boston_P.data

y = Boston_P.target


from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf','linear'],
              'gamma':[1,0.1,0.01]}

grid = GridSearchCV(SVR(),parameters, refit = True, verbose=2, scoring='neg_mean_squared_error')

grid.fit(x,y)

best_params = grid.best_params_


"""""""""""
K-NN Hyper Parameter Testing

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7,
                                                    random_state = 22, shuffle=True, 
                                                    stratify = y)

from sklearn.neighbors import KNeighborsClassifier

KNN_accuracy_test = []
KNN_accuracy_train = []

for k in range(1, 50):
    KNN = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=1)
    KNN.fit(X_train, y_train)
    KNN_accuracy_train.append(KNN.score(X_train, y_train))
    KNN_accuracy_test.append(KNN.score(X_test, y_test))

plt.plot(np.arange(1, 50),KNN_accuracy_train, label="train")
plt.plot(np.arange(1, 50),KNN_accuracy_test, label="test")
plt.xlabel('K')
plt.ylabel('Score')
plt.legend()
plt.show()




















