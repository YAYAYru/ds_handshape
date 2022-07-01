import click
import yaml

import pandas as pd
import numpy as np

from big.src.features.angle import Angle

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def xyz2angle(path_params_yaml: str):
    print("-----------------xyz2angle()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["xyz2angle"]

    path_angle_19f_csv = params_yaml_own["outs"]["path_angle_19f_csv"]
    df = pd.read_csv(params_yaml_own["deps"]["path_xyz_63f_csv"])

    list_column = ["fsw", "videoframe", "filename", "foldername"]
    np_xyz_63f = df.drop(columns=list_column).to_numpy()
    np_xyz_21_3 = np_xyz_63f.reshape(np_xyz_63f.shape[0], int(np_xyz_63f.shape[1]/3), 3)
    a = Angle()
    np_angles = a.xyz2angles_hand(np_xyz_21_3)
    df_angle = pd.DataFrame(np_angles, columns=a.get_list_handkeys())
    
    for n in list_column:
        df_angle[n] = df[n]
    df_angle.to_csv(path_angle_19f_csv, index=False)
    print("angle_19f to csv")


if __name__ == "__main__":
    xyz2angle()