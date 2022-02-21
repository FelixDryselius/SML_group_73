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

    x = film_data.drop(columns=['Lead'] #, 'Year', 'Mean Age Male', 'Mean Age Female', 'Number of male actors', 'Number of female actors', 'Age Co-Lead'])
    y = film_data['Lead']
    positive_class = "Male"
    negative_class = "Female"
    
    

    # Setting values to k-range and number of folds
    k_max = 80
    n_folds = 200
 
    # Call of function "evaluate_with_kfold",
    # please see function def for more info
    missclassification_k_error, evaluation_terms, plotting_terms = evaluate_with_kfold(x, y,positive_class, negative_class, n_folds, k_max)
    
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

def evaluate_with_kfold(x, y, positive_class, negative_class, n_folds=None, k_max=None):
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


    Kfold_cv = skl_ms.KFold(shuffle=True,n_splits=n_folds)
    k_range = np.arange(1,k_max)
    
    # Initializing variables for later usage
    missclassification_k_error = np.zeros(len(k_range))
    evaluation_terms = {}
    plotting_terms = {}
    
    # ----------------- 1st cross validation -----------------
    # Finds the optimal value for 'k' and an estimation of
    # E[Error_new] using cross validation.
    # Number of folds and range of k-values tested are
    # given by n_folds and k_max

    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        
        temp_missclassification_k_error = evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test)
        missclassification_k_error = np.add(missclassification_k_error,temp_missclassification_k_error)
    
    missclassification_k_error /= n_folds
    min_error_k = find_best_k(missclassification_k_error)
    
    # ----------------- 2nd cross validation -----------------
    # Finds values of 'evaluation terms' and 'plotting
    # ranges(terms)' for ROC and precision-recall curve
    # by taking the average of cross validation with n_folds

    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        model = skl_nb.KNeighborsClassifier(n_neighbors=min_error_k)
        model.fit(x_train, y_train)
        
        temp_evaluation_terms, temp_plotting_terms = get_evaluation_terms(model, x_test, y_test, positive_class, negative_class)
        
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
        for key in temp_plotting_terms.keys():
            if key in plotting_terms:
                for i, k in enumerate(temp_plotting_terms[key]):
                    val = plotting_terms[key][i] + temp_plotting_terms[key][i]
                    plotting_terms[key][i]= val
            else:
                # If plotting_terms is empty
                plotting_terms[key] = temp_plotting_terms[key]
        
    for key in plotting_terms.keys():
        # Average of plotting terms over 'n_fold'
        plotting_terms[key]=np.array(plotting_terms[key])/n_folds
    
    # Creation of additional evaluation terms and addition to 
    # dictionary "evaluation_terms"
    evaluation_terms["TPR"] = evaluation_terms["TP"]/evaluation_terms["P"]
    evaluation_terms["FPR"] = evaluation_terms["FP"]/evaluation_terms["N"]
    evaluation_terms["precision"] = evaluation_terms["TP"]/evaluation_terms["P_star"]
    evaluation_terms["F1"] = 2*(evaluation_terms["precision"]*evaluation_terms["TPR"])/(evaluation_terms["precision"]+evaluation_terms["TPR"])  

    return missclassification_k_error, evaluation_terms, plotting_terms
        
        

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


def get_mean_missclassification(model,x_test,y_test):
    # ----------------- function description -----------------
    # This function returns the mean missclassification error 
    # from a 'model' evaluated on 'y_test'

    prediction = model.predict(x_test)
    mean_missclassification =  np.mean(prediction != y_test)
    return mean_missclassification


def get_evaluation_terms(model, x_test, y_test, positive_class, negative_class):
    # ----------------- function description -----------------
    # This function returns the evaluation and plotting terms
    # given a model, test set and class labels.

    prediction = model.predict(x_test)
    predict_prob = model.predict_proba(x_test)
    P = np.sum(y_test == positive_class) #the same as TP+FN
    N = np.sum(y_test == negative_class) #the same as TN+FP 
    
    # All variables with *_curve are lists that contain 
    # plotting values for FPR, TPR, precision and recall.
    # Index represents the value of the threshold 'r' in
    # range 0-1 with 0.01 as increments. 
    FPR_curve = [0]*100
    TPR_curve = [0]*100
    precision_curve = [0]*100 
    recall_curve = [0]*100
    threshold = np.linspace(0,1,101)

    i=0
    positive_class_index = np.argwhere(model.classes_== positive_class).squeeze()
    
    for r in threshold:       
        prediction_curve = np.where(predict_prob[:, positive_class_index]> r, positive_class, negative_class)
        FP_temp = np.sum((prediction_curve == positive_class) & (y_test == negative_class))
        TP_temp = np.sum((prediction_curve == positive_class) & (y_test == positive_class))
        P_star_temp = FP_temp+TP_temp
        
        # Because the value of FP_temp, TP_temp, etc.
        # can be zero, if-statements must be inplace
        # to ensure non-zero division. Because the
        # values of all folds will be combined,
        # zero-valued arrays are acceptable
        if ((TP_temp!=0) and (P_star_temp!=0)):
            val = TP_temp/P_star_temp + precision_curve[i]
            precision_curve[i] = val
        if ((TP_temp!=0) and (P!=0)):
            val = TP_temp/P + recall_curve[i]
            recall_curve[i] = val        
        if ((FP_temp!=0) and (N!=0)):
            val = FP_temp/N + FPR_curve[i]
            FPR_curve[i] = val        
        if ((TP_temp!=0) and (P!=0)):
            val = TP_temp/P + TPR_curve[i]
            TPR_curve[i] = val        
        i +=1
        
    FP = np.sum((prediction == positive_class) & (y_test == negative_class))
    TP = np.sum((prediction == positive_class) & (y_test == positive_class))
    FN = np.sum((prediction == negative_class) & (y_test == positive_class))
    TN = np.sum((prediction == negative_class) & (y_test == negative_class))

    
    P_star = np.sum(prediction == positive_class) #the same as TP+FP
    N_star = np.sum(prediction == negative_class) #the same as TN+F

    evaluation_terms = {"P":P, "N":N,"P_star":P_star, "N_star":N_star, "TN":TN, "FP":FP, "FN":FN, "TP":TP}
    plotting_terms = {"FPR_curve":FPR_curve,"TPR_curve":TPR_curve,"precision_curve":precision_curve,"recall_curve":recall_curve}
    
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



if __name__=="__main__":
    main()

