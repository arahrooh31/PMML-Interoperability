import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#Loading Data
heart_data = pd.read_csv("data/Heart Disease Cleveland.csv")
Y = heart_data['target'].values
X = heart_data.drop(['target'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


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






