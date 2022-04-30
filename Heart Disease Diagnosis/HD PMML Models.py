#Load Dependencies 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
import xml.etree.ElementTree as elt


#Loading Data
heart_data = pd.read_csv('Data/Heart Disease Cleveland.csv')
Y = heart_data['target'].values
X = heart_data.drop(['target'], axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


#Mapping data
mapper = DataFrameMapper([
    (['age'], StandardScaler()), 
    (['sex'], None),
    (['cp'], [CategoricalDomain(), OneHotEncoder(drop='first')]),
    (['trestbps'], StandardScaler()),
    (['chol'], StandardScaler()),
    (['fbs'], None),
    (['restecg'], StandardScaler()),
    (['thalach'], StandardScaler()),
    (['exang'], None),
    (['oldpeak'], StandardScaler()),
    (['slope'], [CategoricalDomain(), OneHotEncoder(drop='first')]),
    (['ca'], StandardScaler()),
    (['thal'], [CategoricalDomain(), OneHotEncoder(drop='first')])
])


#Logistic Regression PMML
LR_pipeline = PMMLPipeline([
    ('mapper', mapper),
    ('classifier', LogisticRegression())
])

LR_pipeline.fit(X_train, Y_train)
mapper_fit = mapper.fit(X_train)
predictions = LR_pipeline.predict(X_test)
LR_precision = metrics.precision_score(predictions, Y_test)
LR_recall = metrics.recall_score(predictions, Y_test)
sklearn2pmml(LR_pipeline, 'Scoring/PMML Files/LogisticRegression_HeartDisease.pmml', with_repr = True)


#Add Performance Metrics to LR PMML
elt.register_namespace('', 'http://www.dmg.org/PMML-4_4')
elt.register_namespace('xmlns:data', 'http://jpmml.org/jpmml-model/InlineTable')

tree = elt.parse('Scoring/PMML Files/LogisticRegression_HeartDisease.pmml')
xmlRoot = tree.getroot()
child = elt.Element('ModelEvaluation')
child.tail = '      \n    '

recall = elt.SubElement(child, 'Recall')
recall.set('recall', str(LR_recall))
recall.tail = '\n       '

recall_calc = elt.SubElement(child, 'RecallCalculation')
recall_calc.set('recall calculation', 'tp / (tp + fn)')
recall_calc.tail = '\n        '

precision = elt.SubElement(child, 'Precision')
precision.set('precision', str(LR_precision))
precision.tail = '\n        '

precision_calc = elt.SubElement(child, 'PrecisionCalculation')
precision_calc.set('precision calculation', 'tp / (tp + fp)')
precision_calc.tail = '\n        '

evaluationmethod = elt.SubElement(child, 'EvaluationMethod')
evaluationmethod.set('Prediction', 'Testing Dataset')
evaluationmethod.tail = '\n'

DataProcessing = elt.SubElement(child, 'DataProcessing')
DataProcessing.set('train/test ', '80/20 split')
DataProcessing.tail = '\n'

xmlRoot.append(child)
elt.dump(xmlRoot)
tree.write('Scoring/PMML Files/LogisticRegression_HeartDisease.pmml', encoding = "UTF-8")


#Decision Tree PMMl
DT_pipeline = PMMLPipeline([
    ('mapper', mapper),
    ('classifier', DecisionTreeClassifier())])
DT_pipeline.fit(X_train, Y_train)
mapper_fit = mapper.fit(X_train)
predictions = DT_pipeline.predict(X_test)
DT_precision = metrics.precision_score(predictions, Y_test)
DT_recall = metrics.recall_score(predictions, Y_test)
sklearn2pmml(DT_pipeline, 'PMML Files/DecisionTree_HeartDisease.pmml', with_repr = True)

#Add Performance Metrics to DT PMML
elt.register_namespace('', 'http://www.dmg.org/PMML-4_4')
elt.register_namespace('xmlns:data', 'http://jpmml.org/jpmml-model/InlineTable')

tree = elt.parse('Scoring/PMML Files/DecisionTree_HeartDisease.pmml')
xmlRoot = tree.getroot()
child = elt.Element('ModelEvaluation')
child.tail = '      \n    '

recall = elt.SubElement(child, 'Recall')
recall.set('recall', str(DT_recall))
recall.tail = '\n       '

recall_calc = elt.SubElement(child, 'RecallCalculation')
recall_calc.set('recall calculation', 'tp / (tp + fn)')
recall_calc.tail = '\n     '  

precision = elt.SubElement(child, 'Precision')
precision.set('precision', str(DT_precision))
precision.tail = '\n        '

precision_calc = elt.SubElement(child, 'PrecisionCalculation')
precision_calc.set('precision calculation', 'tp / (tp + fp)')
precision_calc.tail = '\n    '

evaluationmethod = elt.SubElement(child, 'EvaluationMethod')
evaluationmethod.set('Prediction', 'Testing Dataset')
evaluationmethod.tail = '\n'

DataProcessing = elt.SubElement(child, 'DataProcessing')
DataProcessing.set('train/test ', '80/20 split')
DataProcessing.tail = '\n'

xmlRoot.append(child)
elt.dump(xmlRoot)
tree.write('Scoring/PMML Files/DecisionTree_HeartDisease.pmml')