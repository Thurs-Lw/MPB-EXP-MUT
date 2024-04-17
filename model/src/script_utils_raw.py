from sklearn import metrics
import numpy as np
import pandas as pd


def sensitivity(Y_test,Y_pred,n=2):#n为分类数
    sen = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
    return sen

def specificity(Y_test,Y_pred,n=2):
    spe = []
    con_mat = metrics.confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    return spe

def cal_matrix(true_labels, pred_labels, num_class, load_checkpoint_path, data_file_name=""):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print_result = {"model": load_checkpoint_path.split("/")[-1][:-5], "data": data_file_name}

    print_result["AUC"] = metrics.roc_auc_score(np.eye(num_class)[true_labels], pred_labels, average="macro")
    pred_labels = np.argmax(pred_labels, axis=1)
    print_result["ACC"] = metrics.accuracy_score(true_labels, pred_labels)
    print_result["precision"] = metrics.precision_score(true_labels, pred_labels, average="macro")
    print_result["Recall"] = metrics.recall_score(true_labels, pred_labels, average="macro")
    print_result["F1"] = metrics.f1_score(true_labels, pred_labels, average="macro")
    print_result["MCC"] = metrics.matthews_corrcoef(true_labels, pred_labels)

    if num_class == 2:
        print_result["Sensitivity"] = sensitivity(true_labels, pred_labels, 2)[1]
        print_result["Specificity"] = specificity(true_labels, pred_labels, 2)[1]
        #print(metrics.matthews_corrcoef(true_labels, pred_labels))
        print_result["MCC"] = metrics.matthews_corrcoef(true_labels, pred_labels)

    print("\n========================================")
    print(pd.DataFrame(print_result, index=["model"]))
    print("========================================\n")
    return print_result