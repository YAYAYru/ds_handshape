import click
import yaml

import pandas as pd
import numpy as np

from collections import Counter
# from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import ClusterCentroids, \
                                    AllKNN, \
                                    EditedNearestNeighbours, \
                                    NearMiss, \
                                    RandomUnderSampler


class Df2np2df:
    def df2np(self, df):
        self.column_names = df.columns
        X = df[df.columns.difference(['fsw', 'videoframe', 'filename', 'foldername'])]
        y = df["fsw"].to_numpy()
        return X, y
        
    def np2df(self, X_resampled, y_resampled):
        df_resampled = pd.DataFrame(X_resampled, columns=self.column_names[:-4])
        df_resampled["fsw"] = y_resampled
        return df_resampled


def undersampling_ClusterCentroids(df):
    print("-----------------undersampling_ClusterCentroids()------------------------")
    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    ros = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)

    return df_resampled


def undersampling_AllKNN(df):
    print("-----------------undersampling_AllKNN()------------------------")
    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    ros = AllKNN()
    X_resampled, y_resampled = ros.fit_resample(X, y)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)

    return df_resampled


def undersampling_EditedNearestNeighbours(df):
    print("-----------------undersampling_EditedNearestNeighbours()------------------------")
    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    ros = EditedNearestNeighbours(kind_sel="mode")
    X_resampled, y_resampled = ros.fit_resample(X, y)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)
    return df_resampled


def undersampling_EditedNearestNeighbours_ClusterCentroids(df):
    print("-----------------undersampling_EditedNearestNeighbours_ClusterCentroids()------------------------")

    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    enn = EditedNearestNeighbours(kind_sel="all")
    X_resampled, y_resampled = enn.fit_resample(X, y)

    ros = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)
    return df_resampled


def undersampling_AllKNN_ClusterCentroids(df):
    print("-----------------undersampling_AllKNN_ClusterCentroids()------------------------")

    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(X, y)

    ros = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)
    return df_resampled


def undersampling_NearMiss(df):
    print("-----------------undersampling_NearMiss()------------------------")
    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    ros = NearMiss(version=2)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)
    return df_resampled


def undersampling_RandomUnderSampler(df):
    print("-----------------undersampling_RandomUnderSampler()------------------------")
    df2np2df = Df2np2df()
    X, y = df2np2df.df2np(df)

    ros = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    df_resampled = df2np2df.np2df(X_resampled, y_resampled)
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
    

    if method=="undersampling_ClusterCentroids":
        df = undersampling_ClusterCentroids(df)
    if method=="undersampling_AllKNN":
        df = undersampling_AllKNN(df)
    if method=="undersampling_EditedNearestNeighbours":
        df = undersampling_EditedNearestNeighbours(df)
    if method=="undersampling_EditedNearestNeighbours_ClusterCentroids":
        df = undersampling_EditedNearestNeighbours_ClusterCentroids(df)
    if method=="undersampling_AllKNN_ClusterCentroids":
        df = undersampling_AllKNN_ClusterCentroids(df)
    if method=="undersampling_NearMiss":
        df = undersampling_NearMiss(df)
    if method=="undersampling_RandomUnderSampler":
        df = undersampling_RandomUnderSampler(df)

    c = Counter(df["fsw"])
    print(f'Distribution balancing classes\n: {c}')   
    df.to_csv(path_train_val_test_csv_balance, index=False)


if __name__ == "__main__":
    make_imbalance()