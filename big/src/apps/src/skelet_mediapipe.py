from typing import Tuple

import mediapipe as mp
import numpy as np
import cv2
from numpy import ndarray


class SkeletMediapipe:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.mp_holistic = mp.solutions.holistic

    def video2xyz_all(
        self,
        np_video
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        :return: (133, 33, 3), (133, 21, 3), (133, 21, 3), (133, 466, 3)
        (frame count, point count, x-y-z)
        """
        list_frame_pose = []
        list_frame_righthand = []
        list_frame_lefthand = []
        list_frame_face = []

        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            for image in np_video:
                image.flags.writeable = False

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    list_xyz = []
                    for lm in results.pose_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                else:
                    list_xyz = np.zeros((33, 3)).tolist()
                    # print("Error: a human not found in the image")
                list_frame_pose.append(list_xyz)

                # Потом сделать видео в showskelet.py
                if results.right_hand_landmarks:
                    list_xyz = []
                    for lm in results.right_hand_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                else:
                    list_xyz = np.zeros((21, 3)).tolist()
                    # print("Error: a right hand not found in the image")
                list_frame_righthand.append(list_xyz)

                if results.left_hand_landmarks:
                    list_xyz = []
                    for lm in results.left_hand_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                else:
                    list_xyz = np.zeros((21, 3)).tolist()
                    # print("Error: a left hand not found in the image")
                list_frame_lefthand.append(list_xyz)

                if results.face_landmarks:
                    list_xyz = []
                    for lm in results.face_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                else:
                    list_xyz = np.zeros((468, 3)).tolist()
                    # print("Error: a face not found in the image")
                list_frame_face.append(list_xyz)

        np_frame_pose = np.array(list_frame_pose)
        np_frame_righthand = np.array(list_frame_righthand)
        np_frame_lefthand = np.array(list_frame_lefthand)
        np_frame_face = np.array(list_frame_face)
        return \
            np_frame_pose, np_frame_righthand, np_frame_lefthand, np_frame_face

    def image2xyz_all(self, np_image) -> np.ndarray:
        """
        Сначала найти позу. потом руку и лицо, иначе не может найти руку и лицо
        :return: (33, 3), (21, 3), (21, 3), (476, 3) -> (point count, x-y-z)
        """
        imgRGB = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        with self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True,
        ) as holistic:

            results = holistic.process(imgRGB)

            if results.pose_landmarks:
                list_xyz = []
                for lm in results.pose_landmarks.landmark:
                    list_xyz.append([
                        round(lm.x, 5),
                        round(lm.y, 5),
                        round(lm.z, 5)])
                np_xyz_pose = np.array(list_xyz)
                # print("np_xyz_pose.shape", np_xyz_pose.shape)

                if results.right_hand_landmarks:
                    list_xyz = []
                    for lm in results.right_hand_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                    np_xyz_hand_right = np.array(list_xyz)
                    # print("np_xyz_hand_right.shape", np_xyz_hand_right.shape)
                else:
                    np_xyz_hand_right = np.zeros((21, 3))
                    print("Error: a right hand not found in the image")

                if results.left_hand_landmarks:
                    list_xyz = []
                    for lm in results.left_hand_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                    np_xyz_hand_left = np.array(list_xyz)
                else:
                    np_xyz_hand_left = np.zeros((21, 3))
                    print("Error: a left hand not found in the image")

                if results.face_landmarks:
                    list_xyz = []
                    for lm in results.face_landmarks.landmark:
                        list_xyz.append(
                            [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5)]
                        )
                    np_xyz_face = np.array(list_xyz)
                else:
                    np_xyz_face = np.zeros((478, 3))
                    print("Error: a face not found in the image")

            else:
                np_xyz_pose = np.zeros((33, 3))
                np_xyz_hand_right = np.zeros((21, 3))
                np_xyz_hand_left = np.zeros((21, 3))
                np_xyz_face = np.zeros((478, 3))
                print("Error: a human not found in the image")

        return np_xyz_pose, np_xyz_hand_right, np_xyz_hand_left, np_xyz_face

    def image2xyz_pose(self, np_image) -> np.ndarray:
        """
        :return: (2, 21, 3) -> (hand count, point count, x-y-z)
        """
        imgRGB = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            list_xyz = []
            for lm in results.pose_landmarks.landmark:
                list_xyz.append([
                    round(lm.x, 5),
                    round(lm.y, 5),
                    round(lm.z, 5)])
            np_xyz = np.array(list_xyz)

        else:
            print("Error: a human not found in the image")
            np_xyz = np.array([])

        return np_xyz

    def video2xyz_pose(self, np_video) -> np.ndarray:
        """
        :return: (2, 21, 3) -> (hand count, point count, x-y-z)
        """
        list_frame = []
        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            for image in np_video:
                image.flags.writeable = False

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                list_xyz = []
                for lm in results.pose_landmarks.landmark:
                    list_xyz.append([
                        round(lm.x, 5),
                        round(lm.y, 5),
                        round(lm.z, 5)])

                list_frame.append(list_xyz)
        return np.array(list_frame)

    # TODO find left or right hand?
    # https://toptechboy.com/distinguish-between-right-and-left-hands-in-mediapipe/
    def image2xyz_multihand(self, np_image, hand_count=2) -> np.ndarray:
        """
        :return: (2, 21, 3) -> (hand count, point count, x-y-z)
        """
        imgRGB = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            list_hands = []
            for handLms in results.multi_hand_landmarks[:hand_count]:
                list_xyz = []
                for id, lm in enumerate(handLms.landmark):
                    list_xyz.append([
                        round(lm.x, 5),
                        round(lm.y, 5),
                        round(lm.z, 5)])
                list_hands.append(list_xyz)
            self.hands = np.array(list_hands)
            return self.hands
        else:
            print("Error: a hand not found in the image")

    def video2xyz_multihand(self, np_video, hand_count=2) -> np.ndarray:
        """
        :return: (2, 21, 3) -> (hand count, point count, x-y-z)
        """
        list_frame = []
        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            for image in np_video:
                image.flags.writeable = False

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    list_hands = []
                    for handLms in results.multi_hand_landmarks[:hand_count]:
                        list_xyz = []
                        for id, lm in enumerate(handLms.landmark):
                            list_xyz.append([
                                round(lm.x, 5),
                                round(lm.y, 5),
                                round(lm.z, 5)]
                            )
                        list_hands.append(list_xyz)
                    list_frame.append(list_hands)
        return np.array(list_frame)
