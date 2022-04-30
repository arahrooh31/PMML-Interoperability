#Load Dependencies 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn_pandas import DataFrameMapper
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain
from sklearn2pmml.decoration import ContinuousDomain
from sklearn.preprocessing import StandardScaler
from sklearn2pmml import sklearn2pmml
import xml.etree.ElementTree as elt


#Loading Data
breast_data = pd.read_csv('Data/Breast Cancer Wisconsin.csv', header = 0)
breast_data.drop('Unnamed: 32', axis=1, inplace=True)
breast_data.drop('id', axis=1, inplace=True)
breast_data['diagnosis'] = breast_data['diagnosis'].map({'M':1, 'B':0})
Y = breast_data['diagnosis']
X = breast_data.drop('diagnosis', axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

mapper = DataFrameMapper([
    (['radius_mean'], [ContinuousDomain(), StandardScaler()]),
    (['texture_mean'], [ContinuousDomain(), StandardScaler()]),
    (['perimeter_mean'], [ContinuousDomain(), StandardScaler()]),
    (['area_mean'], [ContinuousDomain(), StandardScaler()]),
    (['smoothness_mean'], [ContinuousDomain(), StandardScaler()]),
    (['compactness_mean'], [ContinuousDomain(), StandardScaler()]),
    (['concavity_mean'], [ContinuousDomain(), StandardScaler()]),
    (['concave points_mean'], [ContinuousDomain(), StandardScaler()]),
    (['symmetry_mean'], [ContinuousDomain(), StandardScaler()]),
    (['fractal_dimension_mean'], [ContinuousDomain(), StandardScaler()]),
    (['radius_se'], [ContinuousDomain(), StandardScaler()]),
    (['texture_se'], [ContinuousDomain(), StandardScaler()]),
    (['perimeter_se'], [ContinuousDomain(), StandardScaler()]),
    (['area_se'], [ContinuousDomain(), StandardScaler()]),
    (['smoothness_se'], [ContinuousDomain(), StandardScaler()]),
    (['compactness_se'], [ContinuousDomain(), StandardScaler()]),
    (['concavity_se'], [ContinuousDomain(), StandardScaler()]),
    (['concave points_se'], [ContinuousDomain(), StandardScaler()]),
    (['symmetry_se'], [ContinuousDomain(), StandardScaler()]),
    (['fractal_dimension_se'], [ContinuousDomain(), StandardScaler()]),
    (['radius_worst'], [ContinuousDomain(), StandardScaler()]),
    (['texture_worst'], [ContinuousDomain(), StandardScaler()]),
    (['perimeter_worst'], [ContinuousDomain(), StandardScaler()]),
    (['area_worst'], [ContinuousDomain(), StandardScaler()]),
    (['smoothness_worst'], [ContinuousDomain(), StandardScaler()]),
    (['compactness_worst'], [ContinuousDomain(), StandardScaler()]),
    (['concavity_worst'], [ContinuousDomain(), StandardScaler()]),
    (['concave points_worst'], [ContinuousDomain(), StandardScaler()]),
    (['symmetry_worst'], [ContinuousDomain(), StandardScaler()]),
    (['fractal_dimension_worst'], [ContinuousDomain(), StandardScaler()]),
])


#Decision Tree To Pmml
DT_pipeline = PMMLPipeline([
    ('mapper', mapper),
    ('classifier', DecisionTreeClassifier())
    ])

DT_pipeline.fit(X_train, Y_train) 
predictions = DT_pipeline.predict(X_test)
DT_precision = metrics.precision_score(predictions, Y_test)
DT_recall = metrics.recall_score(predictions, Y_test)
sklearn2pmml(DT_pipeline, 'PMML Files/DecisionTree_BreastCancer.pmml', with_repr = True)


#Add Performance Metrics to DT PMML
elt.register_namespace('', 'http://www.dmg.org/PMML-4_4')
elt.register_namespace('xmlns:data', 'http://jpmml.org/jpmml-model/InlineTable')

tree = elt.parse('Scoring/PMML Files/DecisionTree_BreastCancer.pmml')
xmlRoot = tree.getroot()
child = elt.Element('ModelEvaluation')
child.tail = '      \n    '

recall = elt.SubElement(child, 'Recall')
recall.set('recall', str(DT_recall))
recall.tail = '\n       '

recall_calc = elt.SubElement(child, 'RecallCalculation')
recall_calc.set('recall calculation', 'tp / (tp + fn)')
recall_calc.tail = '\n   '

precision = elt.SubElement(child, 'Precision')
precision.set('precision', str(DT_precision))
precision.tail = '\n        '

precision_calc = elt.SubElement(child, 'PrecisionCalculation')
precision_calc.set('precision calculation', 'tp / (tp + fp)')
precision_calc.tail = '\n        '

evaluationmethod = elt.SubElement(child, 'EvaluationMethod')
evaluationmethod.set('Prediction', 'Testing Dataset')
evaluationmethod.tail = '\n'

DataProcessing = elt.SubElement(child, 'DataProcessing')
DataProcessing.set('train/test', '70/30 split')
DataProcessing.tail = '\n'
xmlRoot.append(child)
elt.dump(xmlRoot)
tree.write('Scoring/PMML Files/DecisionTree_BreastCancer.pmml')


#Logistic Regression Model to PMML
LR_pipeline = PMMLPipeline([
    ('mapper', mapper),
    ('classifier', LogisticRegression())])

LR_pipeline.fit(X_train, Y_train)
predictions = LR_pipeline.predict(X_test)
LR_precision = metrics.precision_score(predictions, Y_test)
LR_recall = metrics.recall_score(predictions, Y_test)
sklearn2pmml(LR_pipeline, 'Scoring/PMML Files/LogisticRegression_BreastCancer.pmml', with_repr = True)


#Add Performance Metrics to LR PMML
elt.register_namespace('', 'http://www.dmg.org/PMML-4_4')
elt.register_namespace('xmlns:data', 'http://jpmml.org/jpmml-model/InlineTable')

tree = elt.parse('Scoring/PMML Files/LogisticRegression_BreastCancer.pmml')
xmlRoot = tree.getroot()
child = elt.Element('ModelEvaluation')
child.tail = '      \n    '

recall = elt.SubElement(child, 'Recall')
recall.set('recall', str(LR_recall))
recall.tail = '\n       '

recall_calc = elt.SubElement(child, 'RecallCalculation')
recall_calc.set('recall calculation', 'tp / (tp + fn)')
recall_calc.tail = '\n   '

precision = elt.SubElement(child, 'Precision')
precision.set('precision', str(LR_precision))
precision.tail = '\n        '

precision_calc = elt.SubElement(child, 'PrecisionCalculation')
precision_calc.set('precision calculation', 'tp / (tp + fp)')
precision_calc.tail = '\n    '

evaluationmethod = elt.SubElement(child, 'EvaluationMethod')
evaluationmethod.set('Prediction', 'Testing Dataset')
evaluationmethod.tail = '\n'

DataProcessing = elt.SubElement(child, 'DataProcessing')
DataProcessing.set('train/test', '70/30 split')
DataProcessing.tail = '\n'

xmlRoot.append(child)
elt.dump(xmlRoot)
tree.write('Scoring/PMML Files/LogisticRegression_BreastCancer.pmml')