#Load Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

#Loading Data
heart_data = pd.read_csv('Data/Heart Disease Cleveland.csv')
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



#Logistic Regression Hyperparamter Tuning

# Create the model
lrc = LogisticRegression()

# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Define the hyperparameters and their possible values
param_grid = {"C":[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 
              "penalty":['none', 'l1', 'l2', 'elasticnet'],
              "solver":['newton-cg', 'lbfgs', 'liblinear']}


# Create the GridSearchCV object
grid_search = GridSearchCV(lrc, param_grid, scoring='accuracy', n_jobs= -1, cv=cv)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, Y_train)

# Print the best parameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# Explanability with Shap
import shap
masker = shap.maskers.Independent(data = X_test)
model = LogisticRegression(max_iter=10000, C=1, penalty='l2', solver='lbfgs')
model.fit(X_train, Y_train)
explainer = shap.LinearExplainer(model, masker=masker)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
