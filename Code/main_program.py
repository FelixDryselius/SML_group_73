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

    x = film_data.drop(columns=['Lead']) #, 'Year', 'Mean Age Male', 'Mean Age Female', 'Number of male actors', 'Number of female actors', 'Age Co-Lead'])
    y = film_data['Lead']
    positive_class = "Male"
    negative_class = "Female"
    
    

    # Setting values to k-range, number of folds, and threshold
    k_max = 80
    n_folds = 200
    threshold = 0.24
 
    # Call of function "evaluate_with_kfold",
    # please see function def for more info
    missclassification_k_error,  evaluation_terms, plotting_terms = evaluate_with_kfold(x, y,positive_class, negative_class, threshold, n_folds, k_max)
    
    # Finds what k-value minimizes missclassification
    # and find missclassification at that point
    min_error_k = find_best_k(missclassification_k_error)
    new_error_estimate = np.min(missclassification_k_error)
    evaluation_terms["Optimal_k"] = min_error_k
    evaluation_terms["E[Error_new]"] = new_error_estimate

    # Creates list for latex
    data = list(evaluation_terms.items())
    data.insert(0, ["Term","Value"]) 
    data_np = np.array(data)
    print(tabulate(data_np, headers=("firstrow"), tablefmt="latex"))


    # Create the missclassification to 'k' value
    # graph
    plt.plot(np.arange(1,k_max), missclassification_k_error)
    plt.title(f"Cross validation KFold = {n_folds} error for kNN")
    plt.xlabel("k")
    plt.ylabel("Validation error")
    plt.show()
    
    # Create crosstab heatmap
    cm=np.array([ [evaluation_terms["TP"],evaluation_terms["FN"]],\
        [evaluation_terms["FP"],evaluation_terms["TN"]] ])
    disp = met.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
    disp.plot()
    disp.ax_.set_title("Cross tab heatmap")
    plt.show()

    # Create ROC and Recall-Precision graph
    figure, axis = plt.subplots(2)
    figure.set_constrained_layout(True)

    axis[0].plot(plotting_terms["FPR_curve"], plotting_terms["TPR_curve"]);
    axis[0].set_title("ROC curve ")
    axis[0].set_xlabel("False positive rate")
    axis[0].set_ylabel("True positive rate")

    axis[1].plot(plotting_terms["recall_curve"], plotting_terms["precision_curve"]);
    axis[1].set_title("Precision-recall curve")
    axis[1].set_xlabel("Recall rate")
    axis[1].set_ylabel("precision rate")
    plt.show()

def evaluate_with_kfold(x, y, positive_class, negative_class, threshold=0.5, n_folds=None, k_max=None):
    # ----------------- function description -----------------
    # This function has three returns:
    # 'missclassification_k_error', 'evaluation_terms', and
    # 'plotting_terms'. 
    # These are all dictionaries with 'values' estimated
    # by cross validation and 'key' as the name of the 
    # variable or list of plotting points.
    #
    # 'missclassification_k_error' is a np.array where index+1
    # represents k and value at index represents 
    # missclassification error for that k
    # 
    # 'evaluation_terms' is a dictionary containing values
    # of 'evaluation terms', (ex. TP,FP...) where the name
    # of the value is the key
    #
    # 'plotting_terms' is a dictionary containing np.arrays.
    # Each np.array has elements representing the value of 
    # an 'evalueation term' at a certain 'r' threshold. The
    # index represents the 'r' value which ranges from 0 to 1 
    # with increments of 0.01. 
    

    if n_folds==None:
        n_folds=x.shape[0]//4
    if k_max == None:
        k_max= x.shape[1]*4

    # Defining the splits and setting random_state to have reproductable results
    Kfold_cv = skl_ms.KFold(shuffle=True,n_splits=n_folds, random_state=1)
    k_range = np.arange(1,k_max)
    
    # Initializing variables for later usage
    missclassification_k_error = np.zeros(len(k_range))
    evaluation_terms = {}
    threshold_terms = {}
    plotting_terms = {}
    
    # ----------------- 1st cross validation -----------------
    # Finds the optimal value for 'k' and an estimation of
    # E[Error_new] using cross validation with uniform weights.
    # Number of folds and range of k-values tested are
    # given by n_folds and k_max

    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        
        temp_missclassification_k_error = evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test,positive_class, negative_class,"uniform", "auto",threshold)
        missclassification_k_error = np.add(missclassification_k_error,temp_missclassification_k_error)
    
    missclassification_k_error /= n_folds
    min_error_k = find_best_k(missclassification_k_error)
    
    # ----------------- 2nd cross validation -----------------
    # Finds values of 'evaluation terms' and 'plotting
    # ranges(terms)' 

    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        model = skl_nb.KNeighborsClassifier(n_neighbors=min_error_k,weights="uniform",algorithm="auto")
        model.fit(x_train, y_train)
        
        temp_evaluation_terms, temp_threshold_terms = get_evaluation_terms(model, x_test, y_test, positive_class, negative_class, threshold)
        
        # The sum of each 'evaluation terms' generated
        # per fold
        for key in temp_evaluation_terms.keys():
            if key in evaluation_terms:
                evaluation_terms[key] += temp_evaluation_terms[key]
            else: 
                # If evaluation_terms is empty
                evaluation_terms[key] = temp_evaluation_terms[key]

        # The sum of each 'plotting term' per index, per fold
        # This works because all plotting_terms have the same 
        # size 
        for key in temp_threshold_terms.keys():
            if key in threshold_terms:
                for i, k in enumerate(temp_threshold_terms[key]):
                    val = threshold_terms[key][i] + temp_threshold_terms[key][i]
                    threshold_terms[key][i]= val
            else:
                # If plotting_terms is empty
                threshold_terms[key] = temp_threshold_terms[key]
        
  
    
    # Creation of additional evaluation terms and addition to 
    # dictionary "evaluation_terms"
    evaluation_terms["TPR"] = evaluation_terms["TP"]/evaluation_terms["P"]
    evaluation_terms["FPR"] = evaluation_terms["FP"]/evaluation_terms["N"]
    evaluation_terms["accuracy"] = (evaluation_terms["TP"]+evaluation_terms["TN"])/(evaluation_terms["N"]+evaluation_terms["P"])
    evaluation_terms["precision"] = evaluation_terms["TP"]/evaluation_terms["P_star"]
    evaluation_terms["recall"] = evaluation_terms["TP"]/(evaluation_terms["TP"]+evaluation_terms["FN"])
    evaluation_terms["F1"] = 2*(evaluation_terms["precision"]*evaluation_terms["TPR"])/(evaluation_terms["precision"]+evaluation_terms["TPR"])  


    # Creation of specific plotting curves
    plotting_terms["FPR_curve"] = threshold_terms["FP_threshold"]/evaluation_terms["N"]
    plotting_terms["TPR_curve"] = threshold_terms["TP_threshold"]/evaluation_terms["P"]
    plotting_terms["recall_curve"] = threshold_terms["TP_threshold"]/(threshold_terms["FN_threshold"]+threshold_terms["TP_threshold"])
    plotting_terms["precision_curve"] = threshold_terms["TP_threshold"]/(threshold_terms["FP_threshold"]+threshold_terms["TP_threshold"])


    return missclassification_k_error, evaluation_terms, plotting_terms
        
        

def evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test, positive_class, negative_class, weight_type=None, algorithm_type=None, threshold=0.5):
    # ----------------- function description -----------------
    # This function returns the missclassification error 
    # of 'k' in range (1-'k_range').
    # It returns an np.array where index is the value 
    # of ('k'-1) and the value the missclassification error.

    if weight_type == None:
        weight_type = "uniform"
    
    if algorithm_type==None:
        algorithm_type = "auto"    


    missclassification_k_error = np.zeros(len(k_range))
    for index, k in enumerate(k_range):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k,weights=weight_type,algorithm=algorithm_type)
        model.fit(x_train, y_train)
        missclassification_k_error[index] += get_mean_missclassification(model,x_test,y_test, threshold, positive_class, negative_class)
    
    return missclassification_k_error


def get_mean_missclassification(model,x_test,y_test, threshold, positive_class, negative_class):
    # ----------------- function description -----------------
    # This function returns the mean missclassification error 
    # from a 'model' evaluated on 'y_test' and with
    # 'threshold'
    
    # set threshold
    positive_class_index = np.argwhere(model.classes_== positive_class).squeeze()
    prediction = np.where(model.predict_proba(x_test)[:,positive_class_index] > threshold, positive_class, negative_class)
    
    # calc missclasification error
    mean_missclassification =  np.mean(prediction != y_test)
    return mean_missclassification


def get_evaluation_terms(model, x_test, y_test, positive_class, negative_class,threshold):
    # ----------------- function description -----------------
    # This function returns the evaluation and plotting terms
    # given a model, test set and class labels.

    positive_class_index = np.argwhere(model.classes_== positive_class).squeeze()
    
    # Setting based on 'threshold'
    prediction = np.where(model.predict_proba(x_test)[:,positive_class_index] > threshold, positive_class, negative_class)
    prediction = model.predict(x_test)
    predict_prob = model.predict_proba(x_test)
    P = np.sum(y_test == positive_class) #the same as TP+FN
    N = np.sum(y_test == negative_class) #the same as TN+FP 
    
    # All variables with *_threshold are lists that contain 
    # plotting values.
    # Index represents the value of the threshold 'r' in
    # range 0-1 with 0.01 as increments. 
    FP_threshold = np.zeros(101)
    TP_threshold = np.zeros(101)
    FN_threshold = np.zeros(101)
    TN_threshold = np.zeros(101)
    threshold_range = np.linspace(0,1,101)

    i=0    
    for r in threshold_range:       
        prediction_curve = np.where(predict_prob[:, positive_class_index]> r, positive_class, negative_class)
        FP_threshold[i] = np.sum((prediction_curve == positive_class) & (y_test == negative_class))
        TP_threshold[i] = np.sum((prediction_curve == positive_class) & (y_test == positive_class))
        FN_threshold[i] = np.sum((prediction_curve == negative_class) & (y_test == positive_class))
        TN_threshold[i] = np.sum((prediction_curve == negative_class) & (y_test == negative_class))
        i +=1
        
    FP = np.sum((prediction == positive_class) & (y_test == negative_class))
    TP = np.sum((prediction == positive_class) & (y_test == positive_class))
    FN = np.sum((prediction == negative_class) & (y_test == positive_class))
    TN = np.sum((prediction == negative_class) & (y_test == negative_class))

    
    P_star = np.sum(prediction == positive_class) #the same as TP+FP
    N_star = np.sum(prediction == negative_class) #the same as TN+F

    evaluation_terms = {"P":P, "N":N,"P_star":P_star, "N_star":N_star, "TN":TN, "FP":FP, "FN":FN, "TP":TP}
    plotting_terms = {"FP_threshold":FP_threshold,"TP_threshold":TP_threshold,"FN_threshold":FN_threshold,"TN_threshold":TN_threshold}
    
    return evaluation_terms, plotting_terms


def find_best_k( missclassification_k_error):
    # ----------------- function description -----------------
    # This function returns the index+1 of the minimum valued 
    # element in the missclassification_k_error list.
    # This equates to the 'k' value for that point
    #    
    min_error = np.min(missclassification_k_error) 
    min_error_k = [i for i, x in enumerate(missclassification_k_error) if x == min_error] [0]+1

    return min_error_k



# Functions for creating scaled data sets

def generate_standard_scaled_datafile(film_data):
    import sklearn.preprocessing as skl_pre
    #CREATE NEW DATA WITH STANDARDSCALING
    x = film_data.drop(columns=['Lead'])
    y = film_data['Lead']

    standard_scaler = skl_pre.StandardScaler(with_mean=True,with_std=True)

    # StandardScaler: mean=0, variance=1
    scaled_film_data_array = standard_scaler.fit_transform(x)

    x_scaled = pd.DataFrame(scaled_film_data_array, columns = list(x.columns))

    film_data_scaled = x_scaled.join(y)

    film_data_scaled.to_csv('train_standardscaler.csv',index=False)
    
    print("New standardScaled data saved to file: 'train_standardscaler.csv'")

def generate_MinMax_scaled_datafile(film_data):
    import sklearn.preprocessing as skl_pre
    #CREATE NEW DATA WITH MINMAXSCALER
    x = film_data.drop(columns=['Lead'])
    y = film_data['Lead']

    minmax_scaler = skl_pre.MinMaxScaler(feature_range=(0,1))

    # MinMax scaler: min=0, max=1
    scaled_film_data_array = minmax_scaler.fit_transform(x)

    x_scaled = pd.DataFrame(scaled_film_data_array, columns = list(x.columns))

    film_data_scaled = x_scaled.join(y)

    film_data_scaled.to_csv('train_minmaxscaler.csv', index=False)
    
    print("New MinMax data saved to file: 'train_minmaxscaler.csv'")



if __name__=="__main__":
    main()

