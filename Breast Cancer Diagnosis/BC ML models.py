#Load Dependencies 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn_pandas import DataFrameMapper


#Loading Data
breast_data = pd.read_csv("Data/Breast Cancer Wisconsin.csv", header = 0)
breast_data.drop("Unnamed: 32", axis=1, inplace=True)
breast_data.drop("id",axis=1,inplace=True)
breast_data['diagnosis'] = breast_data['diagnosis'].map({'M':1, 'B':0})
Y = breast_data['diagnosis']
X = breast_data.drop('diagnosis', axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


#Decision Tree Model
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, Y_train) 
predictions = DT_model.predict(X_test)
metrics.accuracy_score(predictions, Y_test)
metrics.precision_score(predictions, Y_test)
metrics.recall_score(predictions, Y_test)


#Logistic Regression Model 
LR_model = LogisticRegression()
LR_model.fit(X_train, Y_train) 
predictions = LR_model.predict(X_test)
metrics.accuracy_score(predictions, Y_test)
metrics.precision_score(predictions, Y_test)
metrics.recall_score(predictions, Y_test)