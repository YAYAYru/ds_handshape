import click

import pandas as pd

from showskelet import ShowSkelet

PATH_WORD = "big/data/raw/dict_word_phoneme_v11.csv"

def word_image():
    df = pd.read_csv(PATH_WORD)
    print("df", df)


@click.command()
@click.argument("device_number", type=click.INT)
@click.argument("photo_delay_sec", type=click.INT)
def show_skelet_camera_image_hand(device_number, photo_delay_sec):
    ss = ShowSkelet()
    ss.load_camera_image(device_number, photo_delay_sec)
    ss.process_image2xyz_hand()
    ss.gui_image_hand()
    print(device_number, photo_delay_sec)

if __name__ == "__main__":
    show_skelet_camera_image_hand()