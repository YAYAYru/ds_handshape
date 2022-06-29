import click
import yaml
import json
import glob
import os

import pandas as pd


def json2csv_fsw_signer(path_json_xyz_folders, np_row):
    with open(path_json_xyz_folders + "/" + np_row[0] + ".json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data=data["skelet_frames"])
    assert (np_row[1]!="0" or np_row[2]!=0) and \
           (np_row[1]!=0 or np_row[2]!="0") and\
           (np_row[1]!="0" or np_row[2]!="0") # нет метки, то исправить метки s*****
    if np_row[1]=="0" or np_row[1]==0:
        df["fsw"] = np_row[2]
    else:
        df["fsw"] = np_row[1]

    df["signer"] = os.path.split(np_row[0])[0].split("_")[-1]
    return df

def json2csv_fsw(path_json_xyz_folders, np_row):
    with open(path_json_xyz_folders + "/" + np_row[0] + ".json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data=data["skelet_frames"])
    assert (np_row[1]!="0" or np_row[2]!=0) and \
           (np_row[1]!=0 or np_row[2]!="0") and\
           (np_row[1]!="0" or np_row[2]!="0") # нет метки, то исправить метки s*****
    if np_row[1]=="0" or np_row[1]==0:
        df["fsw"] = np_row[2]
    else:
        df["fsw"] = np_row[1]
    return df

def json2csv_fsw_folder_file(path_json_xyz_folders, np_row):
    df = json2csv_fsw(path_json_xyz_folders, np_row)
    df["videoframe"] = range(0, len(df))
    df["filename"] = os.path.split(np_row[0])[-1]
    df["foldername"] = os.path.split(np_row[0])[0]


    return df    

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def json2csv(path_params_yaml: str):
    print("-----------------json2csv()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["json2csv"]
    params_yaml_video2skelet = params_yaml["video2skelet"]

    path_json_xyz_folders = params_yaml_video2skelet["outs"]["path_json_xyz_folders"]
    path_folder_csv = params_yaml_own["deps"]["path_folder_csv"]

    list_path_json = glob.glob(path_json_xyz_folders +"/*/*.json")
    list_path_csv = glob.glob(path_folder_csv +"/*.csv")

    df = pd.read_csv(list_path_csv[0])
    for n in list_path_csv[1:]:
        df = pd.concat([df, pd.read_csv(n)], ignore_index=True)

    if "foldername" in df:
        df = df.drop(columns=['foldername'])
    np_df = df.to_numpy()


    path = path_json_xyz_folders + "/" + np_df[0,0] + ".json"    
    if os.path.exists(path):
        df = json2csv_fsw_folder_file(path_json_xyz_folders, np_df[0])
    else:
        print("The file not found", path)
    for n in np_df[1:]:
        path = path_json_xyz_folders + "/" + n[0] + ".json"
        if os.path.exists(path):
            df_new = json2csv_fsw_folder_file(path_json_xyz_folders, n)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            print("The file not found", path)       
        
    print("pd.unique(df['fsw']) \n",pd.unique(df["fsw"]))
    # print("pd.unique(df['signer']) \n",pd.unique(df["signer"]))

    df.to_csv(params_yaml_own["outs"]["path_csv"], index=False)
    print("Save path:", params_yaml_own["outs"]["path_csv"])


if __name__ == "__main__":
    json2csv()
