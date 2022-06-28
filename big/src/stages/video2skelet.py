import glob
import os
import click
import json
import yaml

import numpy as np

from big.src.features.skelet_mediapipe import SkeletMediapipe
from big.src.features.workfile import WorkFile


def videofile2skeletjson(path_video: str, path_json: str):
    wf = WorkFile()
    np_video = wf.load_video(path_video)
    sm = SkeletMediapipe()
    np_skelet = sm.video2xyz_multihand(np_video, 1)
    runtime_second = sm.get_runtime("frame")
    if np_skelet.shape[0]==0:
        print("path_video, path_json", path_video, path_json)
        print("Not detect hand")
        return 0
    np_skelet = np_skelet.reshape(np_skelet.shape[0], np_skelet.shape[2], np_skelet.shape[3])
    wf.save_righthand_xyz_np2json(path_json, np_skelet, wf.get_FPS())
    return runtime_second
    

def remove_paths_if_outputpath_exists(np_2paths: np.array):
    list_2paths_for_not_exist = []
    for n in np_2paths:
        if not os.path.exists(n[1]):
            list_2paths_for_not_exist.append(n) 
    np_2paths = np.array(list_2paths_for_not_exist)
    return np_2paths


@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def videofile2skeletjson_folder(path_params_yaml: str):
    print("-----------------video2skelet------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_yaml_own = params_yaml["video2skelet"]

    path_from_folder = params_yaml_own["deps"]["path_video_folders"]
    path_to_folder = params_yaml_own["outs"]["path_json_xyz_folders"]
    # path_reports_video2skelet = params_yaml_own["metrics"]["path_reports_video2skelet"]
    bigdata = params_yaml_own["bigdata"]

    list_path_webm = glob.glob(path_from_folder + "/*/*.webm")
    list_path_MP4 = glob.glob(path_from_folder + "/*/*.MP4")
    list_path_mp4 = glob.glob(path_from_folder + "/*/*.mp4")
    list_path = list_path_webm + list_path_MP4 + list_path_mp4

    if bigdata:
        n = len(list_path)
    else:
        n = 3
    
    print("list_path", list_path[:n])
    print("not os.path.exists(path_to_folder)", not os.path.exists(path_to_folder))
    if not os.path.exists(path_to_folder):
        os.mkdir(path_to_folder)
    
    list_path_folder = []
    for paths in list_path[:n]:
        list_path_folder.append(os.path.dirname(paths)) 
    list_folder = []
    for paths in set(list_path_folder):
        list_folder.append(os.path.split(paths)[-1])

    print("list_folder", list_folder)   

    for path_folder in list_folder:
        path_create_folder = path_to_folder + "/" + path_folder
        if not os.path.exists(path_create_folder):
            os.mkdir(path_create_folder)
    #list_runtime = []

    list_2paths = []
    for paths in list_path[:n]:
        json_path = paths.replace(path_from_folder, path_to_folder) + ".json"
        list_2paths.append([paths, json_path])
    np_2paths = np.array(list_2paths)
    print("np_2paths.shape", np_2paths.shape)
    np_2paths1 = remove_paths_if_outputpath_exists(np_2paths)
    print("np_2paths1.shape", np_2paths1.shape)

    for paths in np_2paths1[:n]:
        # json_path = path.replace(path_from_folder, path_to_folder) + ".json"
        runtime = videofile2skeletjson(paths[0], paths[1])
        # list_runtime.append(runtime)
        print("Transformed from ", paths[0],"to", paths[1], "runtime:", round(runtime, 4), "sec")
    
    """ 
    dict_runtime = {"frame_runtime":round(np.mean(list_runtime), 4)}
    with open(path_reports_video2skelet, "w") as f:
        f.write(json.dumps(dict_runtime))
    """


if __name__ == "__main__":
    videofile2skeletjson_folder()