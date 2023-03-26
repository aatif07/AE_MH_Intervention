import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import figure
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report, confusion_matrix
#import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import fbeta_score
import pickle


train = pd.read_csv('AE_North1.csv')

#Cleaning and replacing
train['Sex'].replace('1', '1', inplace=True)
train['Sex'].replace('2', '2', inplace=True)
train['Sex'].replace('9', '9', inplace=True)
train['Sex'].fillna('1', inplace=True)
train["IMD"].fillna("5", inplace = True)
train["Der_Number_EC_Treatment"].fillna("1", inplace = True)
train["EC_Investigation_01"].fillna("252167001", inplace = True)
train["EC_Treatment_01"].fillna("88140007", inplace = True)
train["Der_Number_EC_Treatment"].fillna("1", inplace = True)
train["AEA_Investigation_01"].fillna("5", inplace = True)
train["AEA_Treatment_01"].fillna("21", inplace = True)
train["Der_Number_AEA_Treatment"].fillna("1", inplace = True)

dfmodel = train[['MH_AandE_Diagnosis',
  'MH_OpenToMHSDSAtTimeOfAEAttendance',
  'IMD',
  'Age',
  'Sex',
  'Ethnic_Category',
  'Age_Band',
  'Der_Number_AEA_Diagnosis',
  'Der_Number_EC_Diagnosis',
  'Der_Number_AEA_Investigation',
  'Der_Number_EC_Investigation',
  'Der_Number_AEA_Treatment',
  'Der_Number_EC_Treatment',
  'EC_Diagnosis_01',
  'EC_Investigation_01',
  'AEA_Investigation_01',
  'EC_Treatment_01',
  'AEA_Treatment_01',
  'HRG_Desc']]
#'Der_Number_EC_Investigation',
 # 'Der_Number_AEA_Treatment',
  #'Der_Number_EC_Treatment',

#Dummy values for model preparation
a = pd.get_dummies(dfmodel['Ethnic_Category'], drop_first=True)
b = pd.get_dummies(dfmodel['IMD'], drop_first=True)
c = pd.get_dummies(dfmodel['Age_Band'], drop_first=True)
d = pd.get_dummies(dfmodel['Sex'], drop_first=True)
e = pd.get_dummies(dfmodel['EC_Investigation_01'], drop_first=True)
f = pd.get_dummies(dfmodel['EC_Treatment_01'], drop_first=True)
g = pd.get_dummies(dfmodel['AEA_Investigation_01'], drop_first=True)
h = pd.get_dummies(dfmodel['HRG_Desc'], drop_first=True)
i = pd.get_dummies(dfmodel['AEA_Treatment_01'], drop_first=True)

dfmodel.drop(['IMD','Sex','Ethnic_Category','Age_Band', 'HRG_Desc','EC_Treatment_01','EC_Investigation_01',
             'AEA_Investigation_01','AEA_Treatment_01'],
             axis=1,inplace=True)

dfmodel = pd.concat([dfmodel,a,b,c,d,h,e,f,g,i],axis=1)

dftest = dfmodel[np.isfinite(dfmodel).all(1)]

X = dftest.drop('MH_OpenToMHSDSAtTimeOfAEAttendance', axis=1)
y = dftest['MH_OpenToMHSDSAtTimeOfAEAttendance']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=101)

#Standardisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
grid = {"n_estimators": [10, 100, 200, 500, 1000, 1200],
"max_depth": [None, 5, 10, 20, 30],
"max_features": ["auto", "sqrt"],
"min_samples_split": [2,4,6],
"min_samples_leaf": [1, 2, 4]}
np.random.seed(42)
clf = RandomForestClassifier(n_jobs=-1)
rs_clf = RandomizedSearchCV(estimator=clf,
param_distributions = grid,
n_iter=30,
cv = 5,
verbose = 2)
rs_clf.fit(X_train, y_train)


pickle.dump(rs_clf, open("model.pkl", "wb"))
