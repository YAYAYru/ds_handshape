import click
import yaml
import json

import numpy as np

from sklearn.preprocessing import LabelEncoder


def label_encoder(name, y):
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    print("label_encoder.classes_", label_encoder.classes_)
    class_count = len(label_encoder.classes_)
    print("class_count", class_count)
    print(name + ".shape", y.shape)
    print("----")
    Y_train_encoder = label_encoder.transform(y)
    print("y_encoder.shape", Y_train_encoder.shape)
    return Y_train_encoder, label_encoder.classes_


@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def to_categorical(path_params_yaml: str):
    print("-----------------to_categorical()------------------------")
    with open(path_params_yaml) as f:
        dvc_yaml = yaml.safe_load(f)  
    dvc_yaml_own = dvc_yaml["to_categorical"]

    path_train_y = dvc_yaml_own["deps"]["path_train_y"]
    path_val_y = dvc_yaml_own["deps"]["path_val_y"]
    path_test_y = dvc_yaml_own["deps"]["path_test_y"]
    path_test2_y = dvc_yaml_own["deps"]["path_test2_y"]

    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    train_y = np.load(path_train_y)
    val_y = np.load(path_val_y) 
    test_y = np.load(path_test_y)     
    test2_y = np.load(path_test2_y)   
    np.load = np_load_old

    train_y_encoder, list_class_train = label_encoder("train_y", train_y)
    val_y_encoder, list_class_val = label_encoder("val_y", val_y)
    test_y_encoder, list_class_test = label_encoder("test_y", test_y)
    test2_y_encoder, list_class_test2 = label_encoder("test2_y", test2_y)

    # Train, val, test, test2 должны совпадать
    assert list_class_train.tolist()==list_class_val.tolist()
    assert list_class_train.tolist()==list_class_test.tolist()
    assert list_class_test.tolist()==list_class_test2.tolist()

    json.dump(
        indent=4,
        obj={"class":list_class_train.tolist()},
        fp=open(dvc_yaml_own["outs"]["path_skelet_hand_f63_json"], 'w')
    )

    np.save(dvc_yaml_own["outs"]["path_train_y_encoder"], train_y_encoder)
    np.save(dvc_yaml_own["outs"]["path_val_y_encoder"], val_y_encoder)
    np.save(dvc_yaml_own["outs"]["path_test_y_encoder"], test_y_encoder)
    np.save(dvc_yaml_own["outs"]["path_test2_y_encoder"], test2_y_encoder)


if __name__ == "__main__":
    to_categorical()