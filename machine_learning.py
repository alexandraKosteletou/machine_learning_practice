import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
!pip install mglearn
import mglearn
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

#load and display data

diabetes = pd.read_csv('C:/Users/30699/Downloads/diabetes_exam.csv')
diabetes.head()

#split data 
#X = all columns except Outcome
#y = the Outcome column (what we're predicting)
                        
X=diabetes.drop('Outcome', axis=1)
y = diabetes['Outcome']

# Split data into:
#70% training
#30% test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

#normalize features

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

#check test set size
math.sqrt(len(y_test))


#Initialize k-NN with:
#k=15 (neighbors)
#Distance metric = Euclidean
#Fit the model using training data

classifier= KNeighborsClassifier(n_neighbors=15, metric='euclidean')
classifier.fit(X_train, y_train)


#Predict outcomes on test data
#Compare predictions (Y_pred) to actual values (y_test)

Y_pred= classifier.predict(X_test)

#Print accuracy (i.e., % of correct predictions)
print(accuracy_score(y_test,Y_pred))

