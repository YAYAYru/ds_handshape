import yaml
import click
import glob
import timeit
import json

from big.src.features.workfile import WorkFile
from big.src.features.skelet_mediapipe import SkeletMediapipe
from big.src.models.nn import Predicter
from big.src.features.angle import Angle

def predict_image(np_image, path_model, path_json_for_model):
    print("np_image", np_image.shape)
    sm = SkeletMediapipe()
    np_skelet = sm.image2xyz_multihand(np_image, 1)
    np_xyz_21_3 = np_skelet.reshape(
        np_skelet.shape[0], 
        np_skelet.shape[1],np_skelet.shape[2]) 
    a = Angle()
    np_angles = a.xyz2angles_hand(np_xyz_21_3)       
    model = Predicter()
    model.load_model(path_model)
    model.load_class_list(path_json_for_model)
    return model.predict_classes(np_angles)


def predict_video(np_video, path_model, path_json_for_model):
    print("np_video", np_video.shape)
    sm = SkeletMediapipe()
    np_skelet = sm.video2xyz_multihand(np_video, 1)
    np_xyz_21_3 = np_skelet.reshape(
        np_skelet.shape[0],
        np_skelet.shape[2], np_skelet.shape[3]
    )
    a = Angle()
    np_angles = a.xyz2angles_hand(np_xyz_21_3)
    model = Predicter()
    model.load_model(path_model)
    model.load_class_list(path_json_for_model)
    print("np_angles.shape", np_angles.shape)
    return model.predict_classes(np_angles)


@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def app(path_params_yaml: str):
    print("----------------app()------------------------")
    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    params_to_categorical = params_yaml["to_categorical"]
    params_train = params_yaml["train"]    
    params_own = params_yaml["app"]
    params_yaml_video2skelet = params_yaml["video2skelet"]
    path_from_folder = params_yaml_video2skelet["deps"]["path_video_folders"]
    list_path = glob.glob(path_from_folder + "/*/*.mp4")
    print("len(list_path)", len(list_path))
    path_model = params_train["outs"]["path_model"]
    path_json_for_model = params_to_categorical["outs"]["path_skelet_hand_f63_json"]
    path_reports_app = params_own["metrics"]["path_reports_app"]
    
    wf = WorkFile()
    i=19
    np_video = wf.load_video(list_path[i])
    print("list_path[i]", list_path[i])
    dict_runtime = {}
    r = 10
    start= timeit.default_timer()
    np_label = predict_video(np_video, path_model, path_json_for_model)
    sec = round((timeit.default_timer() - start)/np_label.shape[0], r)
    dict_runtime["frame_runtime"] = float('{:f}'.format(sec))
    print("np_label", np_label)

    start= timeit.default_timer()
    label = predict_image(np_video[10], path_model, path_json_for_model)
    sec = round((timeit.default_timer() - start), r)
    dict_runtime["image_runtime"] = float('{:f}'.format(sec))
    print("label", label)
    print("dict_runtime", dict_runtime)    

    with open(path_reports_app, "w") as f:
        f.write(json.dumps(dict_runtime))

if __name__ == "__main__":
    app()