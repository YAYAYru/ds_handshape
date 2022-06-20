import click
import yaml
import json
import timeit

import numpy as np

from big.src.models.nn import Predicter

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def predict(path_params_yaml: str):
    print("-----------------predict()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)
    params_select_features = params_yaml["select_features"]  
    params_to_categorical = params_yaml["to_categorical"]
    params_train = params_yaml["train"]    
    params_own = params_yaml["predict"]

    path_model = params_train["outs"]["path_model"]
    path_json_for_model = params_to_categorical["outs"]["path_skelet_hand_f63_json"]
    path_train_x = params_select_features["outs"]["path_train_x"]
    path_val_x = params_select_features["outs"]["path_val_x"]
    path_test_x = params_select_features["outs"]["path_test_x"]

    path_train_y_pred = params_own["outs"]["path_train_y_pred"]
    path_val_y_pred = params_own["outs"]["path_val_y_pred"]
    path_test_y_pred = params_own["outs"]["path_test_y_pred"]
    path_train_y_pred_proba = params_own["outs"]["path_train_y_pred_proba"]
    path_val_y_pred_proba = params_own["outs"]["path_val_y_pred_proba"]
    path_test_y_pred_proba = params_own["outs"]["path_test_y_pred_proba"]    
    
    path_reports_predict = params_own["metrics"]["path_report_predict"]

    #np_load_old = np.load
    #np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    train_x = np.load(path_train_x)
    val_x = np.load(path_val_x)
    test_x = np.load(path_test_x) 
    #np.load = np_load_old
    
    model = Predicter()
    model.load_model(path_model)
    model.load_class_list(path_json_for_model)

    dict_runtime = {}
    r = 10

    start= timeit.default_timer()
    train_y_pred = model.predict_classes(train_x)
    sec = round((timeit.default_timer() - start)/train_x.shape[0], r)
    dict_runtime["mean_train_runtime"] = float('{:f}'.format(sec))
    np.save(path_train_y_pred, train_y_pred)
    print("train_y_pred:", train_y_pred.shape, ", Saved:", path_train_y_pred)

    start= timeit.default_timer()
    val_y_pred = model.predict_classes(val_x) 
    sec = round((timeit.default_timer() - start)/val_x.shape[0], r)
    dict_runtime["mean_val_runtime"] = float('{:f}'.format(sec))
    np.save(path_val_y_pred, val_y_pred)
    print("val_y_pred:", val_y_pred.shape, ", Saved:", path_val_y_pred)

    start= timeit.default_timer()
    test_y_pred = model.predict_classes(test_x)
    sec = round((timeit.default_timer() - start)/test_x.shape[0], r)
    dict_runtime["mean_test_runtime"] = float('{:f}'.format(sec))
    np.save(path_test_y_pred, test_y_pred)
    print("test_y_pred:", test_y_pred.shape, ", Saved:", path_test_y_pred)

    print("dict_runtime", dict_runtime)  

    with open(path_reports_predict, "w") as f:
        f.write(json.dumps(dict_runtime))

    train_y_pred_proba = model.predict_proba(train_x)
    val_y_pred_proba = model.predict_proba(val_x)
    test_y_pred_proba = model.predict_proba(test_x)

    np.save(path_train_y_pred_proba, train_y_pred_proba)
    np.save(path_val_y_pred_proba, val_y_pred_proba)
    np.save(path_test_y_pred_proba, test_y_pred_proba)
    
    """ Для тестирования accuracy и loss
    train_y_encoder = np.load("data/processed/train_y_encoder.npy")
    metric_evaluate = model.evaluate(train_x, train_y_encoder)
    print("metric_evaluate_train", metric_evaluate)

    val_y_encoder = np.load("data/processed/val_y_encoder.npy")
    metric_evaluate = model.evaluate(val_x, val_y_encoder)
    print("metric_evaluate_val", metric_evaluate)

    test_y_encoder = np.load("data/processed/test_y_encoder.npy")
    metric_evaluate = model.evaluate(test_x, test_y_encoder)
    print("metric_evaluate_test", metric_evaluate)
    """

if __name__ == "__main__":
    predict()