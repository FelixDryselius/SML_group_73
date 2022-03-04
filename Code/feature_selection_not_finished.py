# ------------------ kNN evaluation ------------------
# The structure of this program is 

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as met
import os
from tabulate import tabulate


def main():
    # Load data which has already been normalized
    # using StandardScaler() 
    cwd = os.getcwd()
    #URI = (cwd+"\\train.csv")
    URI = (cwd+"\\train_standardscaler.csv")
    #URI = (cwd+"\\train_minmaxscaler.csv")
    film_data = pd.read_csv(URI, dtype={"Lead":str}).dropna().reset_index(drop=True)

    x = film_data.drop(columns=['Lead', 'Year', 'Mean Age Male', 'Mean Age Female', 'Number of male actors', 'Number of female actors', 'Age Co-Lead'])
    y = film_data['Lead']
    n_folds = 5
    k_max = 20

    data = feature_selector(x, y, n_folds=None, k_max=None)
    print(data)

def feature_selector(x, y , n_folds=None, k_max=None):

    if n_folds==None:
        n_folds=x.shape[0]//4
    if k_max == None:
        k_max= x.shape[1]*4


    Kfold_cv = skl_ms.KFold(shuffle=True,n_splits=n_folds)
    k_range = np.arange(1,k_max)
    
    # Initializing variables for later usage
    
    min_error = 1
    features_statistics = {}
    used_columns = []
    i =0
    for p in range(1,x.shape[1]):
        for column_name in (set(x.columns)-set(used_columns)):
            print(column_name)
            missclassification_k_error = np.zeros(len(k_range))

            for train_index, val_index in Kfold_cv.split(x):
                x_train, x_test = x[column_name].iloc[train_index], x[column_name].iloc[val_index]
                y_train, y_test = y.iloc[train_index], y.iloc[val_index]
                
                temp_missclassification_k_error = evaluate_k_kNN(k_range,x_train.to_frame(), y_train, x_test.to_frame(), y_test)
                missclassification_k_error = np.add(missclassification_k_error,temp_missclassification_k_error)
            
            missclassification_k_error /= n_folds
            min_error_temp = find_best_k(missclassification_k_error)

            if min_error_temp< min_error:
                min_error = min_error_temp
                features_statistics = {p:[min_error,column_name]} 
                used_columns[p] = column_name

    return features_statistics



def evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test):
    # ----------------- function description -----------------
    # This function returns the missclassification error 
    # of 'k' in range (1-'k_range').
    # It returns an np.array where index is the value 
    # of ('k'-1) and the value the missclassification error.

    missclassification_k_error = np.zeros(len(k_range))
    for index, k in enumerate(k_range):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        missclassification_k_error[index] += get_mean_missclassification(model,x_test,y_test)

    return missclassification_k_error


def find_best_k( missclassification_k_error):
    # ----------------- function description -----------------
    # This function returns the index+1 of the minimum valued 
    # element in the missclassification_k_error list.
    # This equates to the 'k' value for that point
    #    
    min_error = np.min(missclassification_k_error) 
    min_error_k = [i for i, x in enumerate(missclassification_k_error) if x == min_error] [0]+1

    return min_error_k


def get_mean_missclassification(model,x_test,y_test):
    # ----------------- function description -----------------
    # This function returns the mean missclassification error 
    # from a 'model' evaluated on 'y_test'

    prediction = model.predict(x_test)
    mean_missclassification =  np.mean(prediction != y_test)
    return mean_missclassification

























if __name__=="__main__":
    main()

