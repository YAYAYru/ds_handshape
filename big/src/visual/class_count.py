import click
import yaml

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def class_count(path_params_yaml: str):
    print("----------------class_count()------------------------")
    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)
    params_split_by_folder = params_yaml["split_by_folder"]
    path_train_val_test_csv = params_split_by_folder["outs"]["path_train_val_test_csv"]
    df = pd.read_csv(path_train_val_test_csv)
    ax = sns.countplot(x="fsw", data=df)
    c = Counter(df["fsw"])
    print(f'Distribution before imbalancing: {c}')
    plt.show()
    

if __name__ == "__main__":
    class_count()