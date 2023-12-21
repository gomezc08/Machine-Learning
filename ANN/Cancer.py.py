import numpy as np
import pandas as pd

df = pd.read_csv("MallCustomerDataset_.csv")

x = df.iloc[:, 2:32].values
y = df.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x,y,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()

"""9. Adding the input layer and the first hidden layer"""

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation = "relu", input_dim = 30, units=16,  kernel_initializer = "uniform"))
 # Adding the input layer and the first hidden layer
classifier.add(Dropout(rate=0.1))

"""10. Adding the second hidden layer
"""

# Adding the second hidden layer
classifier.add(Dense(activation = "relu", units=16,  kernel_initializer  = "uniform"))
 # Adding dropout to prevent overfitting
classifier.add(Dropout(rate=0.1))

"""11.  Adding the output layer
"""

# Adding the output layer
classifier.add(Dense(activation = "sigmoid", units=1, kernel_initializer = "uniform"))

"""12. Compining the ANN
"""

# Compining the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

"""13. Fitting the ANN to the Training set"""

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train, batch_size = 100, epochs = 150)

"""
14. Predicting the Test set results
"""

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

"""15. Making the confusion Matrix"""

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# CLASSIFIER 1: DECISION TREE
from sklearn.tree import DecisionTreeClassifier
treeClassifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
treeClassifier.fit(x_train, y_train)

y_pred_tree = treeClassifier.predict(x_test)


# CLASSIFIER 2: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
forest.fit(x_train, y_train)

y_pred_forest = forest.predict(x_test)

# CLASSIFIER 3: NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB()
naiveBayes.fit(x_train, y_train)

y_pred_bayes = naiveBayes.predict(x_test)

# CLASSIFIER 4: KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 6, metric = "minkowski", p = 2)
KNN.fit(x_train, y_train)

y_pred_knn = KNN.predict(x_test)

# CLASSIFER 5: Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

# CLASSIFER 6: Support vector linear
from sklearn.svm import SVC
supportVectorLinear = SVC(kernel = "linear", random_state = 0)
supportVectorLinear.fit(x_train, y_train)

y_pred_svc_linear = supportVectorLinear.predict(x_test) 

# CLASSIFER 7: Support vector non-linear
from sklearn.svm import SVC
supportVector = SVC(kernel = "rbf", random_state = 0)
supportVector.fit(x_train, y_train)

y_pred_svc = supportVector.predict(x_test)

# PREDICTIONS
from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_forest = confusion_matrix(y_test, y_pred_forest)
cm_naive = confusion_matrix(y_test, y_pred_bayes)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svLinear = confusion_matrix(y_test, y_pred_svc_linear)
cm_svRbf = confusion_matrix(y_test, y_pred_svc)

from sklearn.metrics import accuracy_score
acc_ann = accuracy_score(y_test, y_pred)
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_forest = accuracy_score(y_test, y_pred_forest)
acc_naive = acc_tree = accuracy_score(y_test, y_pred_bayes)
acc_knn = acc_tree = accuracy_score(y_test, y_pred_knn)
acc_lr = acc_tree = accuracy_score(y_test, y_pred_lr)
acc_svl = acc_tree = accuracy_score(y_test, y_pred_svc_linear)
acc_svR = acc_tree = accuracy_score(y_test, y_pred_svc)

print("Accuracy of ANN: " + str(acc_ann))
print("Accuracy of Decision tree: " + str(acc_tree))
print("Accuracy of Random Forest: " + str(acc_forest))
print("Accuracy of Naive Bayes: " + str(acc_naive))
print("Accuracy of KNN: " + str(acc_knn))
print("Accuracy of Logistic Regression: " + str(acc_lr))
print("Accuracy of Support Vector Linear Classifier: " + str(acc_svl))
print("Accuracy of Support Vector Non-Linear Classifier: " + str(acc_svR))

"""
It seems that Decision tree as the highest accuracy along with svm linear. 
ANN did well with accruacy but not as well as the others.
"""