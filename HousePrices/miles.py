import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

x = df_train.iloc[:, :-1].values
y = df_train.iloc[:, -1].values

cols_with_missing = [col for col in df_train.columns 
                                 if df_train[col].isnull().any()]                                  
candidate_train_predictors = df_train.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = df_test.drop(['Id'] + cols_with_missing, axis=1)

low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if 
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

x_train = final_train
y_train = y
x_test = final_test

cols_with_missing = [col for col in x_train.columns 
                                 if x_train[col].isnull().any()]
reduced_X_train = x_train.drop(cols_with_missing, axis=1)

reduced_X_test = x_test.drop(cols_with_missing, axis=1)

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,max_features=5, random_state=0)
clf.fit(x_train, y_train)

"""
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(reduced_X_train, y_train)

y_pred = l.predict(reduced_X_test)

#submission = pd.concat([df_test["Id"], pd.Series(final, name='SalePrice')], axis=1)
submission = pd.concat([df_test["Id"], pd.Series(y_pred, name='SalePrice')], axis=1)
submission.to_csv('./submission123.csv', index=False, header=True)
submission
""" 