import numpy as np
import pandas as pd

# read data as df
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
ohe = OneHotEncoder()

# gender...
x[:, 1] = le.fit_transform(x[:, 1])
ct_gender = ColumnTransformer(transformers=[('gender', ohe, [1])], remainder='passthrough') 
x = ct_gender.fit_transform(x)
x = x[:, 1:]
# ever_married
x[:, 6] = le.fit_transform(x[:, 6])
ct_married = ColumnTransformer(transformers=[('ever_married', ohe, [6])], remainder='passthrough') 
x = ct_married.fit_transform(x)
x = x[:, 1:]

# work_type
x[:, 7] = le.fit_transform(x[:, 7])
ct_work = ColumnTransformer(transformers=[('work_type', ohe, [7])], remainder='passthrough') 
x = ct_work.fit_transform(x)
x = x[:, 1:] 

# Residence_type
x[:, 11] = le.fit_transform(x[:, 11])
ct_residence = ColumnTransformer(transformers=[('Residence_type', ohe, [11])], remainder='passthrough') 
x = ct_residence.fit_transform(x)
x = x[:, 1:] 


# smoking_status 
x[:, 14] = le.fit_transform(x[:, 14])
ct_smoke = ColumnTransformer(transformers=[('smoking_status', ohe, [14])], remainder='passthrough') 
x = ct_smoke.fit_transform(x)
x = x[:, 1:] 

# dealing with nan values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:,15:])
x[:,15:] = imputer.transform(x[:,15:])

# split dataset
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size = 0.2, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

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
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_forest = accuracy_score(y_test, y_pred_forest)
acc_naive = acc_tree = accuracy_score(y_test, y_pred_bayes)
acc_knn = acc_tree = accuracy_score(y_test, y_pred_knn)
acc_lr = acc_tree = accuracy_score(y_test, y_pred_lr)
acc_svl = acc_tree = accuracy_score(y_test, y_pred_svc_linear)
acc_svR = acc_tree = accuracy_score(y_test, y_pred_svc)

print("Accuracy of Decision tree: " + str(acc_tree))
print("Accuracy of Random Forest: " + str(acc_forest))
print("Accuracy of Naive Bayes: " + str(acc_naive))
print("Accuracy of KNN: " + str(acc_knn))
print("Accuracy of Logistic Regression: " + str(acc_lr))
print("Accuracy of Support Vector Linear Classifier: " + str(acc_svl))
print("Accuracy of Support Vector Non-Linear Classifier: " + str(acc_svR))