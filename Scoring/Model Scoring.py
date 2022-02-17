from pypmml import Model 
import pandas as pd 
import xml.etree.ElementTree as elt
from bs4 import BeautifulSoup as BS
import csv


LR_HD = "PMML Files/LogisticRegression_HeartDisease.pmml"
LR_BC = "PMML Files/LogisticRegression_BreastCancer.pmml"
DT_HD = "PMML Files/DecisionTree_HeartDisease.pmml"
DT_BC = "PMML Files/DecisionTree_BreastCancer.pmml"


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
    

PMML_files = [LR_HD, LR_BC, DT_HD, DT_BC]

for pmml in PMML_files:
    parsePMML(pmml)

parsePMML(DT_BC)
parsePMML(DT_HD)
parsePMML(LR_BC)
parsePMML(LR_HD)
