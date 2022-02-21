# gridsearch-helper
A simple tool to facilitate use of Scikit-Learn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) across multiple models.

SciKit learn's GridSearchCV is an excellent tool for hyperparameter search, but can only be used with model at a time. Additionally, the dictionary syntax used by the object may be difficult for beginner programmers to understand.
This tool is meant to facillitate model and parameter searching. Users can edit a human-readable config file instead of programmatically assembling parameter dictionaries.

# Installation
Requires:
- sklearn
- pandas
- numpy
- pyyaml

Detailed install instructions may be added at a later date.

# Usage
### Data formatting
You must prepare two files for data input, X and y. Each must be a comma seperated file. X is expected to be a sample x feature matrix with sample rows and feature columns. 
Column names will be ignored. Every column of X is expected to be a feature; X must not contain an index or sample labels. y is expected to be a sample x label matrix or vector. 
y may contain several columns; each column will be treated as a new target against which to train models with X. X and y are expected to be the same number of samples and ordered such that row (i) in X corresponds to row (i) in y.

### Running
Simply run:
`python run.py -X path_to_X.csv -y path_to_y.csv -cv [int] -o path_to_output.csv`

Arguments:
- `X` : Path to the X data csv file.
- `y`: Path to the y data csv file.
- `cv`: Number of folds to be perfomed in [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html).
- `o`: Path to the output result file.

### Config
To adjust which models will be used, edit the `config.yml` file. Write `true` or `false` next to each model to indicate whether to use the model in grid search.

Parameters for each model are found in the `params` section of the config file. To add new params to be searched, append to the list.
For example, to test Random Forest Classifiers using both Gini impurity or entropy as the decision metric, scroll to the `RF` section of params and write:

```
criterion:
 - gini
 - entropy
 ```
 Consult the [Sci Kit Learn](https://scikit-learn.org/) documentation for each model to identify which parameter values are possible.

