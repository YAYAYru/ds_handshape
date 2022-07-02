import click
import yaml

import pandas as pd
import numpy as np

from typing import List

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def select_features(path_params_yaml: str):
    print("-----------------select_features------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["select_features"]

    df_train = pd.read_csv(params_yaml_own["deps"]["path_train_csv"])
    df_val = pd.read_csv(params_yaml_own["deps"]["path_val_csv"])
    df_test = pd.read_csv(params_yaml_own["deps"]["path_test_csv"])
    df_test2 = pd.read_csv(params_yaml_own["deps"]["path_test2_csv"])
    
    # select_droping_features = ["fsw", "videoframe", "filename", "foldername"]
    select_droping_features = ["fsw"]

    np_train_x = df_train.drop(select_droping_features, axis=1).to_numpy()
    path = params_yaml_own["outs"]["path_train_x"]
    np.save(path, np_train_x)
    print("np_train_x.shape", np_train_x.shape, "saved:", path)

    np_val_x = df_val.drop(select_droping_features, axis=1).to_numpy()
    path = params_yaml_own["outs"]["path_val_x"]
    np.save(path, np_val_x)
    print("np_val_x.shape", np_val_x.shape, "saved:", path)

    np_test_x = df_test.drop(select_droping_features, axis=1).to_numpy()
    path = params_yaml_own["outs"]["path_test_x"]
    np.save(path, np_test_x)
    print("np_test_x.shape", np_test_x.shape, "saved:", path)

    select_droping_features = ["fsw", "videoframe", "filename", "foldername"]
    np_test2_x = df_test2.drop(select_droping_features, axis=1).to_numpy()
    path = params_yaml_own["outs"]["path_test2_x"]
    np.save(path, np_test2_x)
    print("np_test2_x.shape", np_test2_x.shape, "saved:", path)


    np_train_y = df_train["fsw"].to_numpy()
    path = params_yaml_own["outs"]["path_train_y"]
    np.save(path, np_train_y)
    print("np_train_y.shape", np_train_y.shape, "saved:", path)

    np_val_y = df_val["fsw"].to_numpy()
    path = params_yaml_own["outs"]["path_val_y"]
    np.save(path, np_val_y)
    print("np_val_y.shape", np_val_y.shape, "saved:", path)

    np_test_y = df_test["fsw"].to_numpy()
    path = params_yaml_own["outs"]["path_test_y"]
    np.save(path, np_test_y)
    print("np_test_y.shape", np_test_y.shape, "saved:", path)

    np_test2_y = df_test2["fsw"].to_numpy()
    path = params_yaml_own["outs"]["path_test2_y"]
    np.save(path, np_test2_y)
    print("np_test2_y.shape", np_test2_y.shape, "saved:", path)

if __name__ == "__main__":
    select_features()