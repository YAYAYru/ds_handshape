import mediapipe as mp
import numpy as np
import cv2
import math


class YoloMediapipe:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.hardframe = (0, 0)

    def set_hardframe(self, wh=(0.06, 0.11)) -> None:
        self.hardframe = wh

    def set_boxframe(self, wh=(1.8, 1.8)) -> None:
        self.boxframe = wh

    def hand_func_part(self, results, hand_count):
        for handLms in results.multi_hand_landmarks[:hand_count]:
            list_xyz = []
            for id, lm in enumerate(handLms.landmark):
                list_xyz.append([
                    round(lm.x, 5),
                    round(lm.y, 5),
                    round(lm.z, 5)])

            x_max = np.array(list_xyz)[:, 0].max()
            y_max = np.array(list_xyz)[:, 1].max()
            x_min = np.array(list_xyz)[:, 0].min()
            y_min = np.array(list_xyz)[:, 1].min()

            x = x_min + (x_max - x_min) / 2
            y = y_min + (y_max - y_min) / 2

            if self.hardframe == (0, 0):
                w = x_max - x
                h = y_max - y
                a = np.array(list_xyz)[0]
                b = np.array(list_xyz)[9]
                distance = math.dist(a, b)
                w = w + w * distance * 1.8
                h = h + w * distance * 1.8
            else:
                w = self.hardframe[0]
                h = self.hardframe[1]
        return [x, y, h, w]

    # TODO find left or right hand?
    # https://toptechboy.com/distinguish-between-right-and-left-hands-in-mediapipe/
    def image2xyhw_multihand(self, np_image, hand_count=2) -> np.ndarray:
        """
        :return: (2, 4) -> (hand count, 2 point:x,y,h,w)
        """

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        imgRGB = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            list_xyhw = []
            xyhw = self.hand_func_part(results, hand_count)
            list_xyhw.append(xyhw)
            self.np_xyhw = np.array(list_xyhw)
            return self.np_xyhw
        else:
            print("Error: a hand not found in the image")

    def video2xyhw_hand(self, np_video, hand_count=2) -> np.ndarray:
        list_xyhw_video = []

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
                    xyhw = self.hand_func_part(results, hand_count)
                    list_xyhw_video.append(xyhw)
        np_xyhw_video = np.array(list_xyhw_video)
        return np_xyhw_video

    def image2image_multihand(self, np_image, hand_count=2) -> np.ndarray:
        h, w, c = np_image.shape
        xyhws = self.image2xyhw_multihand(np_image)
        list_image = []
        for i, xyhw in enumerate(xyhws):
            cx1, cy1, cx2, cy2 = (
                int((xyhw[0] - xyhw[3]) * w),
                int((xyhw[1] - xyhw[2]) * h),
                int((xyhw[0] + xyhw[3]) * w),
                int((xyhw[1] + xyhw[2]) * h),
            )
            original = np_image.copy()
            ROI = original[cy1:cy2, cx1:cx2]
            list_image.append(ROI)
            # cv2.imshow("test" + str(i), ROI)
        np_images = np.array(list_image)
        return np_images

    def imagexyhw2image_multihand(self, np_image, xyhw, hand_count=2) -> np.ndarray:
        h, w, c = np_image.shape
        #xyhws = self.image2xyhw_multihand(np_image)
        xyhws = xyhw.reshape(1,xyhw.shape[0])
        list_image = []
        for i, xyhw in enumerate(xyhws):
            cx1, cy1, cx2, cy2 = (
                int((xyhw[0] - xyhw[3]) * w),
                int((xyhw[1] - xyhw[2]) * h),
                int((xyhw[0] + xyhw[3]) * w),
                int((xyhw[1] + xyhw[2]) * h),
            )
            original = np_image.copy()
            ROI = original[cy1:cy2, cx1:cx2]
            list_image.append(ROI)
            # cv2.imshow("test" + str(i), ROI)
        np_images = np.array(list_image)
        return np_images
    
    def video2video_multihand(
        self, np_video, hand_count=2, resize=(250, 400)
    ) -> np.ndarray:
        list_image = []
        self.np_xyhw_video = self.video2xyhw_hand(np_video, hand_count)
        np_image1 = self.imagexyhw2image_multihand(np_video[0], self.np_xyhw_video[0])[0]
        if self.hardframe!=(0,0):
            resize = (np_image1.shape[0], np_image1.shape[1])
        for i, np_image in enumerate(np_video):
            #print("np_image", np_image.shape)
            #print("self.np_xyhw_video[i]", self.np_xyhw_video[i])
            #BUG
            np_image = self.imagexyhw2image_multihand(np_image, self.np_xyhw_video[i])[0]
            # print("np_image", np_image.shape) - (475, 461, 3)
            if self.hardframe==(0,0):
                np_image = cv2.resize(
                    np_image,
                    dsize=resize,
                    interpolation=cv2.INTER_CUBIC)
            else:
                np_image = cv2.resize(
                    np_image,
                    dsize=resize,
                    interpolation=cv2.INTER_CUBIC)
            list_image.append(np_image)
        np_video = np.array(list_image)
        return np_video
