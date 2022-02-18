#All the imports and data import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms

import os, sys

cwd = os.getcwd()
#URI = (cwd+"\\train.csv")
#URI = (cwd+"\\train_standardscaler.csv")
URI = (cwd+"\\train_minmaxscaler.csv")
film_data = pd.read_csv(URI, dtype={"Lead":str}).dropna().reset_index(drop=True)

x = film_data.drop(columns=['Lead'])
y = film_data['Lead']


#Here I do Kfold cross validation

Kfold_cv= skl_ms.KFold(shuffle=True)

k_range = np.arange(1,80)

missclassification_error = np.zeros(len(k_range))

n_folds = 10

for train_index, val_index in Kfold_cv.split(x):
	x_train, x_val = x.iloc[train_index], x.iloc[val_index]
	y_train, y_val = y.iloc[train_index], y.iloc[val_index]
	
	for index, k in enumerate(k_range):
		model = skl_nb.KNeighborsClassifier(n_neighbors=k)
		model.fit(x_train, y_train)
		prediction = model.predict(x_val)
		missclassification_error[index] += np.mean(prediction != y_val)

missclassification_error /= n_folds
plt.plot(k_range, missclassification_error)
plt.title(f"Cross validation KFold = {n_folds} error for kNN")
plt.xlabel("k")
plt.ylabel("Validation error")
plt.show() 
