import glob
import json

import numpy as np
import pandas as pd

from collections import Counter

path_json_fsw = "big/src/features/fsw_handshape.json"
list_csv_clean = glob.glob("big/data/raw/sl_hand_label/*.csv")
list_csv_dirty = glob.glob("/home/yayay/yayay/git/github/ds_hand/big/data/raw/sl_hand_label_dirty/*.csv")
list_csv_1006 = glob.glob("/home/yayay/yayay/git/github/ds_hand/big/data/raw/1006/*.csv")
list_csv = list_csv_clean + list_csv_dirty + list_csv_1006
df = pd.read_csv(list_csv[0])
for n in list_csv[1:]:
    df = pd.concat([df, pd.read_csv(n)], ignore_index=True)
np_hand = df.to_numpy() # (8946, 4)
print("np_hand.shape", np_hand.shape)

list_fsw = []
list_fsw_shape = []
list_fsw_ori = []
list_fsw_ori_l = []
list_fsw_ori_r = []
for n in np_hand:
    if str(n[-1])!="0":
        list_fsw.append(n[-1,])
    if str(n[-2])!="0":
        list_fsw.append(n[-2])
np_fsw = np.array(list_fsw)
count_1 = 0
for i, n in enumerate(np_fsw):
    n = str(n)
    if n=="":
        print("empty")
        continue
    if n=="-1":
        count_1+=1
        continue
    list_fsw_shape.append(n[:4])
    list_fsw_ori.append(n[4:])

    if n[5] in "01234567":
        list_fsw_ori_r.append(n[4:])

    if n[5] in "89abcdef":
        list_fsw_ori_l.append(n[4:])

np_fsw_shape = np.array(list_fsw_shape)

print("-------------------fsw from files in folders--------------------")
print("count_1", count_1)
print("np_fsw_shape[:5]", np_fsw_shape[:5])
print("np_fsw_shape.shape", np_fsw_shape.shape)
print("len(np.unique(np_fsw_shape))", len(np.unique(np_fsw_shape)))
print("Counter", Counter(np_fsw_shape).most_common())
sort_uniq_shape = sorted(np.unique(np_fsw_shape))
print("sort_uniq_shape", sort_uniq_shape)
print("len(sort_uniq_shape)", len(sort_uniq_shape))

print("-------------------fsw from json--------------------")
f = open(path_json_fsw)
json_fsw = json.load(f)
f.close()
print("len(json_fsw)", len(json_fsw))
list_json_fsw_value = []
for k in json_fsw:
    for v in json_fsw[k]:
        list_json_fsw_value.append(v)
print("list_json_fsw_value", sorted(list_json_fsw_value))     
print("list_value", len(list_json_fsw_value))     
print("set(list_json_fsw_v", sorted(set(list_json_fsw_value)))
print("len(set(list_json_fsw_value))", len(set(list_json_fsw_value)))


 