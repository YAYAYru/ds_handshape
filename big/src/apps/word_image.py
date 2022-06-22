import click

import pandas as pd

from showskelet import ShowSkelet
from big.src.apps.src.view_cv2 import ViewCV2
from big.src.stages.app import predict_image


PATH_WORD = "big/data/raw/dict_word_phoneme_v11.csv"
PATH_MODEL = "big/models/skelet_hand_f19"
    
NAME_FEATURE = ["r_fsw_h1"]

def word_image():
    # df = pd.read_csv(PATH_WORD)
    # print("df", df)
    words = by_phonemes_in_сolumns_to_word_and_filename(["s1ce"], NAME_FEATURE)
    words = words["word"].to_numpy()
    print("words.shape", words.shape)
    print("words", words)

def by_phonemes_in_сolumns_to_word_and_filename(sign_component_values, sign_component_columns):
    #print("sign_component_values, sign_component_columns", sign_component_values, sign_component_columns)
    df = pd.read_csv(PATH_WORD)
    full_columns = ["word","filename"] + sign_component_columns
    df = df[df[sign_component_columns[0]].isin(sign_component_values)][full_columns]
    #print("df1", df[df["word"]=="активный2"])
    for n in sign_component_columns[1:]:
        df = df.append(df[df[n].isin(sign_component_values)][full_columns])
    #print("df2", df[df["word"]=="активный2"])
    df = df.drop_duplicates(subset=["filename"])
    #print("df", df)
    select_index=[]
    if len(sign_component_values)>1:   
        for i, n in enumerate(df.index):
            set_column=set(df[sign_component_columns].values[i].tolist())
            set_values = set(sign_component_values) 
            if len(set_values-set_column)==0:
                select_index.append(n)            
        return df[df.index.isin(select_index)][["word", "filename"]]    
    ret = df[["word", "filename"]]
    # print(sign_component_values, "df: \n", ret)        
    return ret

@click.command()
@click.argument("device_number", type=click.INT)
@click.argument("photo_delay_sec", type=click.INT)
def show_skelet_camera_image_hand(device_number, photo_delay_sec):
    df = pd.read_csv(PATH_WORD)
    # print("df", df)

    ss = ShowSkelet()
    np_image = ss.load_camera_image(device_number, photo_delay_sec)
    print("np_image.shape", np_image.shape)
    label = predict_image(np_image, PATH_MODEL + ".h5", PATH_MODEL + ".json")
    print("label", label)
    df_word = by_phonemes_in_сolumns_to_word_and_filename(label, NAME_FEATURE)
    np_words = df_word["word"].to_numpy()
    print("np_words", np_words)
    view_cv2 = ViewCV2()
    np_image = view_cv2.draw_text(np_image, label)
    view_cv2.gui_image(640, np_image)

if __name__ == "__main__":
    show_skelet_camera_image_hand()
    # word_image()