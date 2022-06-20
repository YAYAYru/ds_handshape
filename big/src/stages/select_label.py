import click
from sqlalchemy import column
import yaml

import pandas as pd
import numpy as np

def select_by_ori(df, list_select_ori):
    df["fsw_ori"] = df["fsw"].map(lambda x: x[4:])
    df = df.loc[df["fsw_ori"].isin(list_select_ori)]
    df = df.drop(columns=["fsw_ori"])
    return df

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def select_label(path_params_yaml: str):
    print("-----------------select_label()------------------------")
    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["select_label"]
    path_fsw_csv =  params_yaml_own["deps"]["path_fsw_csv"]
    list_fsw_orientation = params_yaml_own["list_fsw_orientation"]

    df = pd.read_csv(path_fsw_csv)
    df = select_by_ori(df, list_fsw_orientation)
    print("df", df)
    df["fsw"] = df["fsw"].map(lambda x: x[:-2])
    df.to_csv(params_yaml_own["outs"]["path_fsw_handshape_csv"], index=False)
    print("handshape fsw label to csv")


if __name__ == "__main__":
    select_label()