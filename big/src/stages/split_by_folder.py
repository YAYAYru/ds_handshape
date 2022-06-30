import click
import yaml

import pandas as pd



@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def split_by_folder(path_params_yaml: str):
    print("-----------------split_by_folder()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["split_by_folder"]
    list_foldername_for_test2 = params_yaml_own["list_foldername_for_test2"]
    path_reduce_label_csv = params_yaml_own["deps"]["path_reduce_label_csv"]
    path_test2_csv = params_yaml_own["outs"]["path_test2_csv"]
    path_train_val_test_csv = params_yaml_own["outs"]["path_train_val_test_csv"]
    df = pd.read_csv(path_reduce_label_csv)
    df_test2 = df.loc[df["foldername"].isin(list_foldername_for_test2)]
    df_train_val_test = df.loc[~df["foldername"].isin(list_foldername_for_test2)]

    df_test2.to_csv(path_test2_csv, index=False)
    df_train_val_test.to_csv(path_train_val_test_csv, index=False)


if __name__ == "__main__":
    split_by_folder()
