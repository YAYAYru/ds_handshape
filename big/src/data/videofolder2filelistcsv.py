import os
import pandas as pd

foldername = "video_select_sl_hand_11"
path1 = "data/raw/sl_hand/"
path = path1 + foldername 
csv_path = "data/raw/sl_hand_label/"

file_path_list = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".MP4") or file.endswith(".mp4") or file.endswith(".MOV"):
            # file_path_list.append(os.path.join(root, file))
            filepath = os.path.join(root, file).split("/")[len(os.path.join(root, file).split("/")) - 2] + "/" + \
                       os.path.join(root, file).split("/")[len(os.path.join(root, file).split("/")) - 1]
            # foldername = os.path.join(root, file).split("/")[len(os.path.join(root, file).split("/")) - 2]
            # righthand_classname = os.path.join(root, file).split("\\")[len(os.path.join(root, file).split("\\")) - 1].split("_")[1]
            righthand_classname = os.path.join(root, file).split("/")[len(os.path.join(root, file).split("/")) - 1].split("_")[1]
            file_path_list.append([filepath, righthand_classname, 0])

featurename = ["filepath", "righthand", "lefthand"]
df = pd.DataFrame(file_path_list, columns=featurename)
print("df", df)
print(csv_path+foldername+".csv")
df.to_csv(csv_path+foldername+".csv", index=False)

# print(file_path_list)
print("file count:", len(file_path_list))