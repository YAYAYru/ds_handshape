import click
import yaml

import pandas as pd
import numpy as np

from collections import Counter
# from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids

def undersampling(df):
    column_names = df.columns
    print("column_names", column_names[-5:])    
    ros = ClusterCentroids(random_state=0)

    #X = df.loc[:, df.columns != "fsw"]
    X = df[df.columns.difference(['fsw', 'videoframe', 'filename', 'foldername'])]
    y = df["fsw"].to_numpy()
    
    print("X", X.shape)
    print("y", y.shape)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print("X_resampled", X_resampled.shape)
    print("y_resampled", y_resampled.shape)    

    df_resampled = pd.DataFrame(X_resampled, columns=column_names[:-4])

    df_resampled["fsw"] = y_resampled

    print("df_resampled", df_resampled)
    
    return df_resampled

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def make_imbalance(path_params_yaml: str):
    print("-----------------make_imbalance()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["make_imbalance"]
    path_train_val_test_csv = params_yaml_own["deps"]["path_train_val_test_csv"]
    path_train_val_test_csv_balance = params_yaml_own["outs"]["path_train_val_test_csv_balance"]
    method = params_yaml_own["method"]
    df = pd.read_csv(path_train_val_test_csv)
    c = Counter(df["fsw"])
    print(f'Distribution imbalancing classes\n: {c}')
    

    if method=="undersampling":
        df = undersampling(df)
    
    c = Counter(df["fsw"])
    print(f'Distribution balancing classes\n: {c}')   
    df.to_csv(path_train_val_test_csv_balance, index=False)


if __name__ == "__main__":
    make_imbalance()