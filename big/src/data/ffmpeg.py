import glob
import os

from click import command

PATH_FOLDER_INPUT = "/home/yayay/yayay/git/github/ds_hand/big/data/raw/sl_hand/20220625_AlexeyP_raw/*.MP4"
PATH_FOLDER_OUTPUT = "/home/yayay/yayay/git/github/ds_hand/big/data/raw/sl_hand/20220625_AlexeyP/"

list_path = glob.glob(PATH_FOLDER_INPUT)
print("list_path", list_path)
for n in list_path:
    cmd = "ffmpeg -i " + n +" "+ PATH_FOLDER_OUTPUT + os.path.split(n)[-1]
    print(cmd)
    os.system(cmd)
