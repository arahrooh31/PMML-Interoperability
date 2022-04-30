#Load Dependencies
from pypmml import Model 
import pandas as pd 
import xml.etree.ElementTree as elt
from bs4 import BeautifulSoup as BS
import csv


#Load in PMML Files
LR_HD = 'Scoring/PMML Files/LogisticRegression_HeartDisease.pmml'
LR_BC = 'Scoring/PMML Files/LogisticRegression_BreastCancer.pmml'
DT_HD = 'Scoring/PMML Files/DecisionTree_HeartDisease.pmml'
DT_BC = 'Scoring/PMML Files/DecisionTree_BreastCancer.pmml'


#Parsing through PMML file to extract metadata
def parsePMML(pmml):
    with open(pmml) as fp:
        soup = BS(fp, 'xml')

    model_eval = soup.find_all('ModelEvaluation')
    data_censoring = soup.get('missingValueTreatment')
    data_dictionary = soup.find_all('DataDictionary')

    print('model evaluation: ', model_eval)
    print('data censoring: ', data_censoring)
    print('data dictionary: ', data_dictionary)
    
    try: 
        model_type = soup.TreeModel['functionName']
        algorithm_type = soup.TreeModel['algorithmName']
        print('algorithm type: ', algorithm_type)
        print('model type: ', model_type)
    except:
        model_type = soup.RegressionModel['functionName']
        algorithm_type = soup.RegressionModel['algorithmName']
        print('algorithm type: ', algorithm_type)
        print('model type: ', model_type)
    

#Loop through all PMML files for output visualization
PMML_files = [LR_HD, LR_BC, DT_HD, DT_BC]

for pmml in PMML_files:
    parsePMML(pmml)


#Manual method for testing each file individually
parsePMML(DT_BC)
parsePMML(DT_HD)
parsePMML(LR_BC)
parsePMML(LR_HD)