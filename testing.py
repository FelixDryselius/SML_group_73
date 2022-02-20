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
    
    k_max = 60
    n_folds = 200
 
    missclassification_k_error, evaluation_terms, plotting_terms = evaluate_with_kfold(x, y, n_folds, k_max)
    
    min_error_k = find_best_k(missclassification_k_error)
    new_error_estimate = np.min(missclassification_k_error)
    evaluation_terms["Optimal_k"] = min_error_k
    evaluation_terms["E[Error_new]"] = new_error_estimate
    
  
    data = list(evaluation_terms.items())
    data.insert(0, ["Term","Value"]) 
    data_np = np.array(data)
    print(tabulate(data_np, headers=("firstrow")))

    figure, axis = plt.subplots(2, 2)
    figure.set_constrained_layout(True)

    axis[0,0].plot(np.arange(1,k_max), missclassification_k_error)
    axis[0,0].set_title(f"Cross validation KFold = {n_folds} error for kNN")
    axis[0,0].set_xlabel("k")
    axis[0,0].set_ylabel("Validation error")
    
    axis[0,1].plot(plotting_terms["FPR_curve"], plotting_terms["TPR_curve"]);
    axis[0,1].set_title("ROC curve ")
    axis[0,1].set_xlabel("False positive rate")
    axis[0,1].set_ylabel("True positive rate")

    axis[1,0].plot(plotting_terms["recall_curve"], plotting_terms["precision_curve"]);
    axis[1,0].set_title("Precision-recall curve")
    axis[1,0].set_xlabel("Recall rate")
    axis[1,0].set_ylabel("precision rate")

    cm=np.array([ [evaluation_terms["TP"],evaluation_terms["FN"]],\
        [evaluation_terms["FP"],evaluation_terms["TN"]] ])
    disp = met.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
    #disp.title("Cross tab heatmap")
    disp.plot(ax=axis[1,1])
    disp.ax_.set_title("Cross tab heatmap")
        #plt.xlabel("Recall rate")
    #plt.ylabel("precision rate")
    plt.show()

def evaluate_with_kfold(x, y, n_folds=None, k_max=None):
    if n_folds==None:
        n_folds=x.shape[0]//4
    if k_max == None:
        k_max= x.shape[1]*4


    Kfold_cv = skl_ms.KFold(shuffle=True,n_splits=n_folds)
    k_range = np.arange(1,k_max)
    missclassification_k_error = np.zeros(len(k_range))
    evaluation_terms = {}
    plotting_terms = {}
    
    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        
        temp_missclassification_k_error = evaluate_k_kNN(k_range,x_train, y_train, x_test, y_test)
        missclassification_k_error = np.add(missclassification_k_error,temp_missclassification_k_error)
    
    missclassification_k_error /= n_folds
    min_error_k = find_best_k(missclassification_k_error)

    for train_index, val_index in Kfold_cv.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]
        model = skl_nb.KNeighborsClassifier(n_neighbors=min_error_k)
        model.fit(x_train, y_train)
        
        temp_evaluation_terms, temp_plotting_terms = get_evaluation_terms(model, x_test, y_test)
        for key in temp_evaluation_terms.keys():
            if key in evaluation_terms:
                evaluation_terms[key] += temp_evaluation_terms[key]
            else:
                evaluation_terms[key] = temp_evaluation_terms[key]

        for key in temp_plotting_terms.keys():
            if key in plotting_terms:
                #plotting_terms[key] = np.add(plotting_terms[key],temp_plotting_terms[key])
                plotting_terms[key] = [sum(x) for x in zip(plotting_terms[key], temp_plotting_terms[key])]
            else:
                plotting_terms[key] = temp_plotting_terms[key]

    for key in plotting_terms.keys():
        plotting_terms[key]=np.array(plotting_terms[key])/n_folds

    evaluation_terms["TPR"] = evaluation_terms["TP"]/evaluation_terms["P"]
    evaluation_terms["FPR"] = evaluation_terms["FP"]/evaluation_terms["N"]
    evaluation_terms["precision"] = evaluation_terms["TP"]/evaluation_terms["P_star"]

    return missclassification_k_error, evaluation_terms, plotting_terms
        
        


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


def get_evaluation_terms(model, x_test, y_test):
    positive_class ="Male" #y_test.value_counts().idxmax()
    negative_class= "Female"#y_test.value_counts().idxmin()
    
    prediction = model.predict(x_test)
    predict_prob = model.predict_proba(x_test)
    P = np.sum(y_test == positive_class) #the same as TP+FN
    N = np.sum(y_test == negative_class) #the same as TN+FP 
    
    FPR_curve =[]
    TPR_curve = []
    precision_curve = [] 
    recall_curve = []



    #FPR_curve, TPR_curve, threshold = met.roc_curve(y_test,  predict_prob, pos_label=positive_class )
    #precision_curve, recall_curve, threshold = met.precision_recall_curve(y_test,  predict_prob, pos_label=positive_class )
    
    threshold = np.linspace(0,1,101)
    positive_class_index = np.argwhere(model.classes_== positive_class).squeeze()
    for r in threshold:
        prediction_curve = np.where(predict_prob[:, positive_class_index]> r, positive_class, negative_class)
        FP_temp = np.sum((prediction_curve == positive_class) & (y_test == negative_class))
        TP_temp = np.sum((prediction_curve == positive_class) & (y_test == positive_class))
        P_star_temp = FP_temp+TP_temp
        FPR_curve.append(FP_temp/N)
        TPR_curve.append(TP_temp/P)
        
        if P_star_temp == 0:
            continue
        else:
            precision_curve.append(TP_temp/P_star_temp)
            recall_curve.append(TP_temp/P)     
        
    FP = np.sum((prediction == positive_class) & (y_test == negative_class))
    TP = np.sum((prediction == positive_class) & (y_test == positive_class))
    FN = np.sum((prediction == negative_class) & (y_test == positive_class))
    TN = np.sum((prediction == negative_class) & (y_test == negative_class))
    
 
    #TP = met.confusion_matrix(y_test,prediction).ravel()
    P_star = np.sum(prediction == positive_class) #the same as TP+FP
    N_star = np.sum(prediction == negative_class) #the same as TN+F


    #auc_roc = met.auc(FPR_curve,TPR_curve)
    #auc_precision_recall = met.auc(precision_curve,recall_curve)
    #auc_precision_recall = met.average_precision_score(y_test, predict_prob, pos_label="Male")

    evaluation_terms = {"P":P, "N":N,"P_star":P_star, "N_star":N_star, "TN":TN, "FP":FP, "FN":FN, "TP":TP,\
        "auc_roc":5,"auc_precision_recall":5}
    plotting_terms = {"FPR_curve":FPR_curve,"TPR_curve":TPR_curve,"precision_curve":precision_curve,"recall_curve":recall_curve}
    
    return evaluation_terms, plotting_terms


def find_best_k( missclassification_k_error):
    min_error = np.min(missclassification_k_error) 
    min_error_k = [i for i, x in enumerate(missclassification_k_error) if x == min_error] [0]+1

    return min_error_k








def flatten(dic, prefix = ""):
    if prefix != "":
        prefix = prefix + "."

    result = {}

    for k, v in dic.items():
        if isinstance(v, type(dict)):
            for k1, v1 in flatten(v, prefix + k).items():
                result[k1] = v1
        else:
            result[prefix + k] = v

    return result


if __name__=="__main__":
    main()
