import click
import yaml
import json

import pandas as pd


def data_pre_validation(np_fsw, dict_reduce_label):
    print("-----------------data_pre_validation()------------------------")
    set_fsw = set(np_fsw)
    print("len(set_fsw)", len(set_fsw))
    list_value = []
    for n in dict_reduce_label.keys():
        list_value = list_value + dict_reduce_label[n]

    len_list_value = len(list_value)
    set_value = set(list_value)
    len_set_value = len(set_value)
    print("len_list_value", len_list_value)
    print("len_set_value", len_set_value)
    assert len_list_value==len_set_value

    s = set_fsw - set_value
    len_s = len(s)
    if len_s:
        print(s, "are/is missing in the json file")
        assert len_s==0
    else:
        print("Enough!!!")


def data_post_validation(np_fsw, dict_reduce_label):
    print("-----------------data_post_validation()------------------------")
    set_fsw = set(np_fsw)
    #print("set_fsw", set_fsw)
    set_key = set(dict_reduce_label.keys())
    #print("set_key", set_key)
    len_fsw = len(set_fsw)
    len_key = len(set_key)
    s = set_fsw ^ set_key
    print(s, "are/is missing in the dataset(csv)")
    print("len(s)", len(s))
    print("len_fsw", len_fsw, "==", len_key, "len_key")
    assert len_fsw==len_key


def reducelabel(np_fsw, dict_reduce_label):
    data_pre_validation(np_fsw, dict_reduce_label)

    s = 0
    break_for_1 = 0
    
    for i, fsw in enumerate(np_fsw):
        for key in dict_reduce_label.keys():
            for value in dict_reduce_label[key]:
                s+=1
                """                
                print("s", s)
                print("fsw", fsw)
                print("key", key)
                print("value", value)
                print("------------------")
                """
                if fsw==value:
                    np_fsw[i] = key
                    break_for_1 = 1
                    break

            if break_for_1==1:
               break_for_1 = 0
               break
    data_post_validation(np_fsw, dict_reduce_label)
    return np_fsw
    

@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def reduce_label(path_params_yaml: str):
    print("-----------------reduce_label()------------------------")
    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["reduce_label"]
    path_reduce_label_json =  params_yaml_own["deps"]["path_reduce_label_json"]
    path_fsw_handshape_csv =  params_yaml_own["deps"]["path_fsw_handshape_csv"]
    path_reduce_label_csv =  params_yaml_own["outs"]["path_reduce_label_csv"]

    df = pd.read_csv(path_fsw_handshape_csv)

    f = open(path_reduce_label_json)
    dict_reduce_label = json.load(f)
    np_fsw = df["fsw"].to_numpy()
    
    np_fsw = reducelabel(np_fsw, dict_reduce_label)
    df["fsw"] = np_fsw
    df.to_csv(path_reduce_label_csv, index=False)

if __name__ == "__main__":
    reduce_label()