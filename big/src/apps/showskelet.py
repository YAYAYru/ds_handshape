from src.workdevice import WorkDevice
from src.workfile import WorkFile
from src.skelet_mediapipe import SkeletMediapipe
from src.view_cv2 import ViewCV2


class ShowSkelet:
    def __init__(self) -> None:
        self.__view = ViewCV2()

    def load_camera_video(
        self,
        device=0,
        first_recording_frame=0,
        last_record_sec=0
    ):
        wd = WorkDevice(device)
        self.__video = wd.load_video(first_recording_frame, last_record_sec)
        return self.__video

    def load_camera_image(self, device, photo_delay):
        wd = WorkDevice(device)
        self.__image = wd.load_image(photo_delay)
        return self.__image

    def load_image(self, path):
        wf = WorkFile()
        self.__image = wf.load_image(path)
        return self.__image

    def load_video(self, path):
        wf = WorkFile()
        self.__video = wf.load_video(path)
        self.__fps = wf.get_FPS()
        return self.__video

    def process_image2xyz_hand(self):
        sm = SkeletMediapipe()
        self.__np_xyz_hand_image = sm.image2xyz_multihand(self.__image, 1)
        return self.__np_xyz_hand_image

    def process_image2xyz(self):
        sm = SkeletMediapipe()
        # self.__np_xyz_pose_image = sm.image2xyz_pose(self.__image)
        (
            self.__np_xyz_pose_image,
            self.__np_xyz_hand_left_image,
            self.__np_xyz_hand_right_image,
            _,
        ) = sm.image2xyz_all(self.__image)

    def process_video2xyz(self):
        sm = SkeletMediapipe()
        # self.__np_xyz_hand_video = sm.video2xyz_multihand(self.__video)
        # self.__np_xyz_pose_video = sm.video2xyz_pose(self.__video)
        (
            self.__np_xyz_pose_video,
            self.__np_xyz_hand_left_video,
            self.__np_xyz_hand_right_video,
            _,
        ) = sm.video2xyz_all(self.__video)

    def gui_image_hand(self):
        if self.__np_xyz_hand_image is not None:
            self.__view.draw_image_skelet_multihand(
                self.__image, self.__np_xyz_hand_image
            )
        self.__view.gui_image(1080, self.__image)

    def gui_image(self):
        self.__view.draw_image_skelet_hand(
            self.__image,
            self.__np_xyz_hand_left_image)
        self.__view.draw_image_skelet_hand(
            self.__image,
            self.__np_xyz_hand_right_image)
        self.__view.draw_image_skelet_pose(
            self.__image,
            self.__np_xyz_pose_image)
        self.__view.gui_image(1080)
    
    def gui_image_fsw(self):
        self.__view.gui_image(1080)

    def gui_video(self):
        self.__view.draw_video_skelet_hand(
            self.__video,
            self.__np_xyz_hand_left_video)
        self.__view.draw_video_skelet_hand(
            self.__video,
            self.__np_xyz_hand_right_video)
        self.__view.draw_video_skelet_pose(
            self.__video,
            self.__np_xyz_pose_video)
        self.__view.gui_video(1080, self.__fps)

    def main_show_skelet_video(self, file_name):
        self.load_video(file_name)
        self.process_video2xyz()
        self.gui_video()
