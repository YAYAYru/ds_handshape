from sqlalchemy import true
import yaml
import click
import json
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from pycm import *

from sklearn.metrics import classification_report, accuracy_score, log_loss

def classification_report_df(dist_report):
    d = {}
    name_col = ["precision", "recall", "f1-score", "support"]
    list_keys = list(dist_report.keys())
    list_keys.remove('accuracy')
    d["class"] = list_keys
    for nc in name_col:
        d[nc] = []
        for n in list_keys:
            if "accuracy"==n:
                continue
            di = dist_report[n][nc]
            if isinstance(di, float):
                d[nc].append(round(di, 4))
            else:
                d[nc].append(di)
    return pd.DataFrame(d) 

def plot_confusion_matrix_v2(confmat, path_image=""):
    print("confmat", confmat.shape)
    ticks=np.linspace(0, confmat.shape[0]-1,num=confmat.shape[0])
    plt.imshow(confmat)
    plt.colorbar()
    plt.xticks(ticks,fontsize=confmat.shape[0])
    plt.yticks(ticks,fontsize=confmat.shape[0])
    plt.grid(True)
    # plt.show()
    if not path_image=="":
        plt.savefig(path_image)

def plot_confusion_matrix_v1(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.inferno_r, path_image=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    if not path_image=="":
        plt.savefig(path_image)

    #plt.cla()
    return plt.figure()


@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def evaluate(path_params_yaml: str):
    print("-----------------evaluate()------------------------")
    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_own = params_yaml["evaluate"]
    params_predict = params_yaml["predict"]
    params_select_features = params_yaml["select_features"]

    path_train_y = params_select_features["outs"]["path_train_y"]
    path_val_y = params_select_features["outs"]["path_val_y"]    
    path_test_y = params_select_features["outs"]["path_test_y"]

    path_train_y_pred = params_predict["outs"]["path_train_y_pred"]
    path_val_y_pred = params_predict["outs"]["path_val_y_pred"]
    path_test_y_pred = params_predict["outs"]["path_test_y_pred"]

    path_train_y_pred_proba = params_predict["outs"]["path_train_y_pred_proba"]
    path_val_y_pred_proba = params_predict["outs"]["path_val_y_pred_proba"]
    path_test_y_pred_proba = params_predict["outs"]["path_test_y_pred_proba"]    

    path_report_train = params_own["metrics"]["path_report_train"]
    path_report_val = params_own["metrics"]["path_report_val"]
    path_report_test = params_own["metrics"]["path_report_test"]

    path_ytrain_ypred = params_own["plots"]["path_ytrain_ypred"]
    path_yval_ypred = params_own["plots"]["path_yval_ypred"]
    path_ytest_ypred = params_own["plots"]["path_ytest_ypred"]
    
    path_train_classreport = params_own["plots"]["path_classreport_train"]
    path_val_classreport = params_own["plots"]["path_classreport_val"]
    path_test_classreport = params_own["plots"]["path_classreport_test"]

    path_confusion_matrix_train = params_own["plots"]["path_confusion_matrix_train"]
    path_confusion_matrix_val = params_own["plots"]["path_confusion_matrix_val"]
    path_confusion_matrix_test = params_own["plots"]["path_confusion_matrix_test"]    

    path_confusion_matrix_norm_train = params_own["plots"]["path_confusion_matrix_norm_train"]
    path_confusion_matrix_norm_val = params_own["plots"]["path_confusion_matrix_norm_val"]
    path_confusion_matrix_norm_test = params_own["plots"]["path_confusion_matrix_norm_test"]    

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    train_y = np.load(path_train_y)
    val_y = np.load(path_val_y)
    test_y = np.load(path_test_y)
    train_y_pred = np.load(path_train_y_pred)
    val_y_pred = np.load(path_val_y_pred)
    test_y_pred = np.load(path_test_y_pred)
    np.load = np_load_old

    metrics_train_dict = classification_report(train_y, train_y_pred, output_dict=True) 
    metrics_val_dict = classification_report(val_y, val_y_pred, output_dict=True) 
    metrics_test_dict = classification_report(test_y, test_y_pred, output_dict=True)

    df_train = classification_report_df(metrics_train_dict)
    df_val = classification_report_df(metrics_val_dict)
    df_test = classification_report_df(metrics_test_dict)
    
    df_train.to_csv(path_train_classreport, index=False)
    df_val.to_csv(path_val_classreport, index=False)
    df_test.to_csv(path_test_classreport, index=False)

    df_train = pd.DataFrame({"train": train_y, "pred": train_y_pred})
    df_train.to_csv(path_ytrain_ypred, index=False)
    df_val = pd.DataFrame({"val": val_y, "pred": val_y_pred})
    df_val.to_csv(path_yval_ypred, index=False)
    df_test = pd.DataFrame({"test": test_y, "pred": test_y_pred})
    df_test.to_csv(path_ytest_ypred, index=False)

    cm = ConfusionMatrix(actual_vector=train_y, predict_vector=train_y_pred) # Create CM From Data
    cm.save_html(path_confusion_matrix_train)
    cm.save_html(path_confusion_matrix_norm_train, normalize=True)
    
    cm = ConfusionMatrix(actual_vector=val_y, predict_vector=val_y_pred) # Create CM From Data
    cm.save_html(path_confusion_matrix_val)
    cm.save_html(path_confusion_matrix_norm_val, normalize=True)
    
    cm = ConfusionMatrix(actual_vector=test_y, predict_vector=test_y_pred) # Create CM From Data
    cm.save_html(path_confusion_matrix_test)
    cm.save_html(path_confusion_matrix_norm_test, normalize=True)
    
    r = 4
    acc_train = round(accuracy_score(train_y, train_y_pred), r)
    train_y_pred_proba = np.load(path_train_y_pred_proba)
    loss_train = round(log_loss(train_y, train_y_pred_proba), r)
    dict_train = {"loss": loss_train, "acc": acc_train}
    with open(path_report_train, "w") as f:
        f.write(json.dumps(dict_train))

    acc_val = round(accuracy_score(val_y, val_y_pred), r)
    val_y_pred_proba = np.load(path_val_y_pred_proba)
    loss_val = round(log_loss(val_y, val_y_pred_proba), r)
    dict_val = {"loss": loss_val, "acc": acc_val}
    with open(path_report_val, "w") as f:
        f.write(json.dumps(dict_val))


    acc_test = round(accuracy_score(test_y, test_y_pred),r)
    test_y_pred_proba = np.load(path_test_y_pred_proba)
    loss_test = round(log_loss(test_y, test_y_pred_proba), r)
    dict_test = {"loss": loss_test, "acc": acc_test}
    with open(path_report_test, "w") as f:
        f.write(json.dumps(dict_test))


if __name__ == "__main__":
    evaluate()