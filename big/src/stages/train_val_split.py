import click
import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from typing import List


# python3 src/stages/train_val_split.py data/processed/train_val.csv data/processed/train.csv data/processed/val.csv
@click.command()
#@click.argument("from_csv", type=click.Path(exists=True))
#@click.argument("to_csvs", type=click.Path(), nargs=2)
#def train_val_split(from_csv: str, to_csvs: List[str]):
@click.argument("path_params_yaml", type=click.Path(exists=True))
def train_val_split(path_params_yaml: str):
    print("-----------------train_val_split------------------------")
    # https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
    with open(path_params_yaml) as conf_file:
        params_yaml = yaml.safe_load(conf_file)  
    params_yaml_own = params_yaml["train_val_split"]
    params_yaml_split_trainval_test = params_yaml["split_trainval_test"]

    path_train_val_csv = params_yaml_split_trainval_test["outs"]["path_train_val_csv"]
    
    df = pd.read_csv(path_train_val_csv)
    df_train, df_val = train_test_split(
        df, 
        test_size=params_yaml_own["val_size"], 
        random_state=params_yaml_own["random_state"])

    path = params_yaml_own["outs"]["path_train_csv"]
    print("Saved: ", path, ", row count:", len(df_train.index))
    df_train.to_csv(path, index=False)

    path = params_yaml_own["outs"]["path_val_csv"]
    print("Saved: ", path, ", row count:", len(df_val.index))
    df_val.to_csv(path, index=False)


if __name__ == "__main__":
    train_val_split()