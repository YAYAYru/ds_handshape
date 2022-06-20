import click
import yaml

import pandas as pd

from sklearn.model_selection import train_test_split
from typing import List


@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def split_trainval_test(path_params_yaml: str):
    print("-----------------split_trainval_test()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["split_trainval_test"]
    
    
    
    
    path_trainval_csv = params_yaml_own["deps"]["path_angle_19f_handshape_csv"]
    
    df = pd.read_csv(path_trainval_csv)
    df_trainval, df_test = train_test_split(
        df, 
        test_size=params_yaml_own["test_size"], 
        random_state=params_yaml_own["random_state"])

    path = params_yaml_own["outs"]["path_train_val_csv"]
    print("Saved: ", path, ", row count:", len(df_trainval.index))
    df_trainval.to_csv(path, index=False)

    path = params_yaml_own["outs"]["path_test_csv"]
    print("Saved: ", path, ", row count:", len(df_test.index))
    df_test.to_csv(path, index=False)










    
    """
    df = pd.read_csv(params_yaml_own["deps"]["path_angle_19f_csv"])
    df_train_val = df.loc[df['signer'].isin(params_yaml_own["train_val"])]
    df_test = df.loc[df['signer'].isin(params_yaml_own["test"])]
    path = params_yaml_own["outs"]["path_train_val_csv"]
    print("Saved: ", path, ", by signer:", params_yaml_own["train_val"])
    df_train_val.to_csv(path, index=False)
    path = params_yaml_own["outs"]["path_test_csv"]
    print("Saved: ", path, ", by signer:", params_yaml_own["test"])
    df_test.to_csv(path, index=False)
    """

if __name__ == "__main__":
    split_trainval_test()
