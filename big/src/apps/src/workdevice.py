import cv2
import numpy as np

from src.work_abc import WorkABC


class WorkDevice(WorkABC):
    def __init__(self, device) -> None:
        self.__device = device

    # For write image
    # https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
    # for key pause
    # https://stackoverflow.com/questions/59529277/how-to-pause-videostream-while-holding-down-a-key-python-opencv
    def load_image(self, photo_delay) -> np.ndarray:

        cap = cv2.VideoCapture(self.__device)
        cv2.namedWindow("Load image")
        img_counter = 0
        photo = False
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            cv2.imshow("Load image", image)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

            if k % 256 == 32:
                photo = True

            if photo:
                img_counter += 1

            if img_counter > photo_delay * fps:
                # SPACE pressed
                print("Assigned image {}".format(image.shape))
                return image

        cap.release()
        cv2.destroyAllWindows()

    def load_video(self, first_record_sec, last_record_sec) -> np.ndarray:

        cap = cv2.VideoCapture(self.__device)
        cv2.namedWindow("Load video")

        record = False
        list_video = []

        # Manual Recording Management
        if first_record_sec == 0 and last_record_sec == 0:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # cv2.imshow("Load video", cv2.flip(image, 1))
                cv2.imshow("Load video", image)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                if k % 256 == 32:
                    # SPACE pressed
                    # print("Assigned video {}".format(image.shape))
                    # return image
                    if record:
                        np_video = np.array(list_video)
                        print("Assigned video {}".format(np_video.shape))
                        return np_video
                    record = not record
                if record:
                    list_video.append(image)

        fps = cap.get(cv2.CAP_PROP_FPS)
        img_counter = 0
        # Records controlled by specified parameters
        # (first_record_sec, last_record_sec)
        if last_record_sec > 0 and last_record_sec > first_record_sec:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                # cv2.imshow("Load video", cv2.flip(image, 1))
                cv2.imshow("Load video", image)

                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                # if k%256 == 32:

                if record:
                    np_video = np.array(list_video)
                    print("Assigned video {}".format(np_video.shape))
                    return np_video

                if (
                    img_counter >= first_record_sec * fps
                    and img_counter <= last_record_sec * fps
                ):
                    list_video.append(image)
                if img_counter >= last_record_sec * fps:
                    record = not record

                img_counter += 1

        cap.release()
        cv2.destroyAllWindows()
