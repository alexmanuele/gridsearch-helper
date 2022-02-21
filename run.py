from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import numpy as np
import itertools as it
import yaml
import argparse

from gridsearch.pipelinehelper import PipelineHelper
from formatter.efl_data_formatter import *

model_dict = {
    'LR': LogisticRegression,
    'NB': MultinomialNB,
    'RF': RandomForestClassifier,
    'GB': GradientBoostingClassifier,
    'SVM': LinearSVC
}

parser = argparse.ArgumentParser(description='Run gridsearch, passing X and y tables/')
parser.add_argument('-X', type=str, help='Filepath to csv file containing feature data. Sample rows, feature columns, comma seperated.')
parser.add_argument('-y', type=str, help='Filepath to csv file containing label data. Sample rows, label columns. May contain more than one label.')
parser.add_argument('-cv', type=int, help='Number of folds for cross validation.')
parser.add_argument('-o', type=str, help='output file path')

if __name__ == '__main__':
    # Load the config file
    with open('conf/config.yml', 'r') as f:
        config = yaml.load(f)

    # parse the arguments and load the data
    args = parser.parse_args()
    Xdf = pd.read_table(args.X, sep=',')
    ydf = pd.read_table(args.y, sep=',')
    cv = args.cv

    try:
        assert Xdf.shape[0] == ydf.shape[0]
    except:
        raise ValueError('X and y do not have the same number of samples!')

    #Get the models indicated as to be used from config
    models_used = [k for k,v in config['use_models'].items() if v]
    models = {k:v() for k, v in model_dict.items() if k in models_used}

    # Get the params for the selected models
    params = {k:v for k,v in config['params'].items() if k in models_used}

    # Initiate the gridsearch
    results = []
    #Each column in y is expected to be a target.
    for col in ydf:
        gridsearch = PipelineHelper(models, params)
        gridsearch.fit(Xdf.values, ydf[col].values, cv=cv)
        result = gridsearch.score_summary()
        result['target'] = col
        results.append(result)

    pd.concat(results).to_csv(args.o, sep=',', index=False)
    
