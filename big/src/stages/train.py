import click
import yaml
import json

import pandas as pd
import numpy as np

from typing import List
from tensorflow.keras import datasets, layers, models, utils, losses

def model_nn(X_train_shape_1, class_count):
    print("-----------------model()------------------------")
    model = models.Sequential()
    model.add(layers.Dense(44, activation='relu', input_shape=(X_train_shape_1,)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(class_count, activation="softmax"))
    model.summary()
    return model

def compile_fit(model, X_train, Y_train_encoder, X_val, Y_val_encoder, epochs):
    print("-----------------compile_fit()------------------------")
    model.compile(optimizer="adam",
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(X_train, Y_train_encoder, epochs=epochs, 
                        validation_data=(X_val, Y_val_encoder),verbose=2)
    return history


# python3 src/stages/train.py data/processed/train.csv data/processed/val.csv models/f63.h5
@click.command()
@click.argument("path_params_yaml", type=click.Path(exists=True))
def train(path_params_yaml: str):
    print("-----------------train()------------------------")

    with open(path_params_yaml) as f:
        params_yaml = yaml.safe_load(f)  
    select_features = params_yaml["select_features"]
    to_categorical = params_yaml["to_categorical"]
    params_yaml_own = params_yaml["train"]

    epochs = params_yaml_own["epochs"]
    path_train_x = select_features["outs"]["path_train_x"]
    path_val_x = select_features["outs"]["path_val_x"]
    path_train_y_encoder = to_categorical["outs"]["path_train_y_encoder"]
    path_val_y_encoder = to_categorical["outs"]["path_val_y_encoder"]
    path_model = params_yaml_own["outs"]["path_model"]
    path_history = params_yaml_own["plots"]["path_history"]


    train_x = np.load(path_train_x)
    val_x = np.load(path_val_x)  
    train_y_encoder = np.load(path_train_y_encoder)
    val_y_encoder = np.load(path_val_y_encoder)      
    
    model = model_nn(train_x.shape[1], len(np.unique(train_y_encoder)))
    history = compile_fit(model, \
        train_x, train_y_encoder, \
        val_x, val_y_encoder, \
        epochs)
    model.save(path_model)

    """ Для тестирования accuracy и loss
    path_report_train = params_yaml_own["metrics"]["path_report_train"]
    path_report_val = params_yaml_own["metrics"]["path_report_val"]   
    train_metrics_dict = model.evaluate(
        train_x,
        train_y_encoder,
        return_dict=True)
    with open(path_report_train, "w") as f:
        f.write(json.dumps(train_metrics_dict))
    val_metrics_dict = model.evaluate(
        val_x,
        val_y_encoder,
        return_dict=True,)
    with open(path_report_val, "w") as f:
        f.write(json.dumps(val_metrics_dict))
    """

    print("history.params", history.params)
    df = pd.DataFrame(history.history)
    df["epoch"] = range(epochs)
    df = df.reindex(columns=["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
    print("df\n", df)
    df.to_csv(path_history, index=False) 


if __name__ == "__main__":
    train()