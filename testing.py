#All the imports and data import
"""




import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da



"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as met
import os
from tabulate import tabulate


def main():
    cwd = os.getcwd()
    #URI = (cwd+"\\train.csv")
    URI = (cwd+"\\train_standardscaler.csv")
    #URI = (cwd+"\\train_minmaxscaler.csv")
    film_data = pd.read_csv(URI, dtype={"Lead":str}).dropna().reset_index(drop=True)

    x = film_data.drop(columns=['Lead'])
    y = film_data['Lead']
    
    k_max = 50
    n_folds = 150
 
    missclassification_k_error = evaluate_with_kfold(x, y, n_folds, k_max)
    
    plt.plot(np.arange(1,k_max), missclassification_k_error)
    plt.title(f"Cross validation KFold = {n_folds} error for kNN")
    plt.xlabel("k")
    plt.ylabel("Validation error")
    plt.show() 


    min_error_k = find_best_k(missclassification_k_error)


    print_table = [["Term","Value"],
                    ["'Postive class'", "Male"],
                    ["Optimal k", min_error_k],
                    ["E[error_new]", 0,1]]

    print(tabulate(print_table, headers=("firstrow")))




def evaluate_with_kfold(x, y, n_folds=None, k_max=None):
    if n_folds==None:
        n_folds=x.shape[0]//4
    if k_max == None:
        k_max= x.shape[1]*4


    Kfold_cv = skl_ms.KFold(shuffle=True,n_splits=n_folds)
    k_range = np.arange(1,k_max)
    missclassification_k_error = np.zeros(len(k_range))
    
    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        
        temp_missclassification_k_error = evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test)
        missclassification_k_error = np.add(missclassification_k_error,temp_missclassification_k_error)
    
    missclassification_k_error /= n_folds
    
    return missclassification_k_error
        
        


def evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test):
    missclassification_k_error = np.zeros(len(k_range))
    for index, k in enumerate(k_range):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        missclassification_k_error[index] += get_mean_missclassification(model,x_test,y_test)
    
    return missclassification_k_error


def get_mean_missclassification(model,x_test,y_test):
    prediction = model.predict(x_test)
    mean_missclassification =  np.mean(prediction != y_test)
    return mean_missclassification


def get_evaluation_terms(model, x_test, y_test, positive_class, negative_class):
    prediction = model.predict(x_test)
    predict_prob = model.predict_proba(x_test)[::,1]
    P = np.sum(y_test == positive_class)
    N = np.sum(y_test == negative_class)
    TN, FP, FN, TP = met.confusion_matrix(y_test,prediction).ravel()
    return P, N, TN, FP, FN, TP


def find_best_k( missclassification_k_error):
    min_error = np.min(missclassification_k_error)
    min_error_k = [i for i, x in enumerate(missclassification_k_error) if x == min_error] [0]+1

    return min_error_k



if __name__=="__main__":
    main()
