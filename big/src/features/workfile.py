import cv2
import numpy as np
import json
import pandas as pd
import glob
import os


class WorkFile:
    def to_video_np(self):
        print("FileWorker.to_video_np", self.path)

    def load_image(self, path: str) -> np.ndarray:
        if ".png" or ".jpg" in path:
            cap = cv2.VideoCapture(path)
            success, self.__image = cap.read()
            if not success:
                print("Can't receive image (stream end?). Exiting ...")
            return self.__image
        else:
            print("Error:", path.split(".")[-1], "is not image file format")

    def load_video(self, path: str) -> np.ndarray:
        if ".mp4" or ".mov" in path:
            self.__cap = cv2.VideoCapture(path)
            list_frame = []
            while self.__cap.isOpened():
                success, image = self.__cap.read()
                if not success:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                list_frame.append(image)
            self.__video = np.array(list_frame)
            return self.__video
        else:
            print("Error:", path.split(".")[-1], "is not video file format")

    def get_FPS(self) -> int:
        return self.__cap.get(cv2.CAP_PROP_FPS)

    
    def save_righthand_xyz_np2json(self, path_json, np_skelet, fps):
        dict_skelet = {"repo_link":"https://github.com/YAYAYru/slsru_ml/"}
        dict_skelet["version"] = "0.5.0.1" 
        dict_skelet["fps"] = fps
        dict_skelet["skelet_frames"] = {} 
        list_frame = []
        str_name = "rhand_"
        for n in range(np_skelet.shape[1]):
            dict_skelet["skelet_frames"][str_name + "x" + str(n)] = []
            dict_skelet["skelet_frames"][str_name + "y" + str(n)] = []
            dict_skelet["skelet_frames"][str_name + "z" + str(n)] = []

        for frame_i, frame in enumerate(np_skelet):
            list_frame.append(frame_i)
            for xyz_i, xyz in enumerate(frame):
                dict_skelet["skelet_frames"][str_name + "x" + str(xyz_i)].append(xyz[0])
                dict_skelet["skelet_frames"][str_name + "y" + str(xyz_i)].append(xyz[1])
                dict_skelet["skelet_frames"][str_name + "z" + str(xyz_i)].append(xyz[2])

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        # TODO without indent for list
        json_object = json.dumps(dict_skelet, default=np_encoder)
        with open(path_json, "w") as outfile:
            outfile.write(json_object)


    def save_righthand_angle_np2json(self, path_json, np_skelet, handkeys, fps):
        dict_skelet = {"repo_link":"https://github.com/YAYAYru/slsru_ml/"}
        dict_skelet["version"] = "0.7.0" 
        dict_skelet["fps"] = fps
        dict_skelet["skelet_angle_frames"] = {} 
        list_frame = []
        for i in range(np_skelet.shape[1]):
            dict_skelet["skelet_angle_frames"][handkeys[i]] = []
        for frame_i, frame in enumerate(np_skelet):
            # list_frame.append(frame_i)
            for xyz_i, xyz in enumerate(frame):
                dict_skelet["skelet_angle_frames"][handkeys[xyz_i]].append(xyz)

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()

        json_object = json.dumps(dict_skelet, default=np_encoder)
        with open(path_json, "w") as outfile:
            outfile.write(json_object)


    def load_righthand_xyz_json2df_signer_label(self, path_json):
        f = open(path_json)
        data = json.load(f)
        df = pd.DataFrame(data=data["skelet_frames"])
        df["fsw"]=os.path.split(path_json)[1].split("_")[1]
        df["signer"]=os.path.split(path_json)[1].split("_")[2]
        return df


    def load_righthand_xyz_json2df_signer_label_from_folder(self, path_from_folder):
        list_path = glob.glob(path_from_folder +"/*.json")
        df = self.load_righthand_xyz_json2df_signer_label(list_path[0])
        for n in list_path[1:]:
            # df.append(self.load_righthand_xyz_json2df(n), ignore_index=True)
            df = pd.concat([df, self.load_righthand_xyz_json2df_signer_label(n)], ignore_index=True)
        return df


    def load_righthand_xyz_json2np(self, path_json):
        f = open(path_json)
        data = json.load(f)
        fps = data["fps"]

        list_frame = []
        count = int(len(data["skelet_frames"].keys())/3)
        for i in range(len(data["skelet_frames"]["rhand_x0"])):
            list_hand_point = []
            for j in range(count):
                x = data["skelet_frames"]["rhand_x"+str(j)][i]
                y = data["skelet_frames"]["rhand_y"+str(j)][i]
                z = data["skelet_frames"]["rhand_z"+str(j)][i]

                list_hand_point.append([x, y, z])
            list_frame.append(list_hand_point)
        return np.array(list_frame), fps

    def load_righthand_angle_json2np(self, path_json):
        f = open(path_json)
        data = json.load(f)
        fps = data["fps"]

        list_frame = []
        count = int(len(data["skelet_angle_frames"].keys()))
        list_keys = list(data["skelet_angle_frames"].keys())

        for i in range(len(data["skelet_angle_frames"][list_keys[0]])):
            list_angle = []
            for j in range(count):
                list_angle.append(data["skelet_angle_frames"][list_keys[j]][i])
            list_frame.append(list_angle)
        return np.array(list_frame), fps

    def save_video_np2npy(self, np_video, file_path_to):
        np.save(file_path_to + ".npy", np_video)

    def load_video_npy2np(self, file_path_npy):
        return np.load(file_path_npy)