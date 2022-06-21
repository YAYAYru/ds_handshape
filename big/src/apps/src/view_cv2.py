import cv2
import numpy as np
import time
from screeninfo import get_monitors

from src.view_abc import View


class ViewCV2(View):
    def __init__(self) -> None:
        super().__init__()

    def draw_video_skelet_multihand(self, video, xyz_objects_frames):
        if xyz_objects_frames is None:
            return video
        list_image = []

        for image, xyz_objects in zip(video, xyz_objects_frames):
            image = self.draw_image_skelet_multihand(image, xyz_objects)
            list_image.append(image)

        self.video = np.array(list_image)
        return video

    def draw_video_skelet_hand(self, video, xyz_object_frames):
        if xyz_object_frames is None:
            return video
        list_image = []

        for image, xyz_object in zip(video, xyz_object_frames):
            image = self.draw_image_skelet_hand(image, xyz_object)
            list_image.append(image)

        self.video = np.array(list_image)
        return video

    def draw_image_skelet_hand(self, image, xyz_object):
        if xyz_object is None:
            self.image = image
            return image

        h, w, c = image.shape
        for line_point in self.list_line_point_hand:
            cx1, cy1, cx2, cy2 = (
                int(xyz_object[line_point[0]][0] * w),
                int(xyz_object[line_point[0]][1] * h),
                int(xyz_object[line_point[1]][0] * w),
                int(xyz_object[line_point[1]][1] * h),
            )
            cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 4)
        for xyz in xyz_object:
            cx, cy = int(xyz[0] * w), int(xyz[1] * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        self.image = image
        return image

    def draw_image_skelet_multihand(self, image, xyz_objects):
        if xyz_objects is None:
            self.image = image
            return image

        h, w, c = image.shape

        for xyz_object in xyz_objects:
            for line_point in self.list_line_point_hand:
                cx1, cy1, cx2, cy2 = (
                    int(xyz_object[line_point[0]][0] * w),
                    int(xyz_object[line_point[0]][1] * h),
                    int(xyz_object[line_point[1]][0] * w),
                    int(xyz_object[line_point[1]][1] * h),
                )
                cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 4)
            for xyz in xyz_object:
                cx, cy = int(xyz[0] * w), int(xyz[1] * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        self.image = image
        return image

    def draw_image_yolo_multihand(self, image, xy_objects):
        if xy_objects is None:
            self.image = image
            return image

        h, w, c = image.shape

        for xyz_object in xy_objects:
            cx, cy = int(xyz_object[0] * w), int(xyz_object[1] * h)
            cv2.circle(image, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

            cx1, cy1, cx2, cy2 = (
                int((xyz_object[0] - xyz_object[3]) * w),
                int((xyz_object[1] - xyz_object[2]) * h),
                int((xyz_object[0] + xyz_object[3]) * w),
                int((xyz_object[1] + xyz_object[2]) * h),
            )
            start = (cx1, cy1)
            end = (cx2, cy2)
            color = (0, 255, 225)
            cv2.rectangle(image, start, end, color, 4)

        self.image = image
        return image

    def draw_video_yolo_hand(self, video, xyhw_object_frames):
        if xyhw_object_frames is None:
            return video

        list_image = []
        for image, xyhw_object in zip(video, xyhw_object_frames):
            xyhw_object = xyhw_object.reshape(1, xyhw_object.shape[0])
            image = self.draw_image_yolo_multihand(image, xyhw_object)
            list_image.append(image)

        self.video = np.array(list_image)
        return self.video

    def draw_video_skelet_angle_multihand(
        self,
        video,
        xyz_object_frames,
        angles
    ):
        if xyz_object_frames is None:
            return video

        list_image = []

        for image, xyz_object, angle in zip(video, xyz_object_frames, angles):
            image = self.draw_image_skelet_angle_multihand(
                image,
                xyz_object,
                angle)
            list_image.append(image)

        self.video = np.array(list_image)
        return self.video

    def draw_image_skelet_angle_multihand(self, image, xyz_objects, angles):
        if xyz_objects is None:
            self.image = image
            return image

        list_for_zero_is_not_angle = [4, 20]
        z = 0
        for n in list_for_zero_is_not_angle:
            angles = np.insert(angles, n, z)

        h, w, c = image.shape

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)

        for xyz_object in xyz_objects:
            for line_point in self.list_line_point_hand:
                cx1, cy1, cx2, cy2 = (
                    int(xyz_object[line_point[0]][0] * w),
                    int(xyz_object[line_point[0]][1] * h),
                    int(xyz_object[line_point[1]][0] * w),
                    int(xyz_object[line_point[1]][1] * h),
                )
                cv2.line(image, (cx1, cy1), (cx2, cy2), (0, 255, 0), 4)
            for i, xyz in enumerate(xyz_object):
                cx, cy = int(xyz[0] * w), int(xyz[1] * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(image, str(angles[i]), (cx, cy), font, 1, color)

        self.image = image
        return image

    def draw_video_skelet_pose(self, video, xyz_object_frames):
        if xyz_object_frames is None:
            return video

        list_image = []

        for image, xyz_object in zip(video, xyz_object_frames):
            image = self.draw_image_skelet_pose(image, xyz_object)
            list_image.append(image)

        self.video = np.array(list_image)
        return self.video

    def draw_image_skelet_pose(self, image, xyz_object):
        if xyz_object is None:
            return image

        h, w, c = image.shape
        list_line_point_pose = [
            [15, 13],
            [13, 11],
            [11, 12],
            [12, 14],
            [14, 16],
            [11, 23],
            [23, 24],
            [12, 24],
        ]

        for line_point in list_line_point_pose:
            cx1, cy1, cx2, cy2 = (
                int(xyz_object[line_point[0]][0] * w),
                int(xyz_object[line_point[0]][1] * h),
                int(xyz_object[line_point[1]][0] * w),
                int(xyz_object[line_point[1]][1] * h),
            )
            cv2.line(image, (cx1, cy1), (cx2, cy2), (255, 255, 0), 4)
        for xyz in xyz_object:
            cx, cy = int(xyz[0] * w), int(xyz[1] * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        self.image = image
        return image

    def draw_text(self, image, text):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (200, 50)
        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 2
        image = cv2.putText(image, text[0], org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        return image

    def draw_list_text(self, image, texts):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (10, 10)
        # fontScale
        fontScale = 0.5
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness = 1
        for i, n in enumerate(texts):
            if i==0:
                org = (10, 20)
            else:
                org = (10, 20*i)
            image = cv2.putText(image, n, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        return image

    def gui_image(self, width, image=None):
        if image is None:
            image_resize = self.ResizeWithAspectRatio(self.image, width=width)
        else:
            image_resize = self.ResizeWithAspectRatio(image, width=width)
        cv2.imshow("Image", image_resize)
        cv2.waitKey(0)

    def ResizeWithAspectRatio(
        self, image, width=None, height=None, inter=cv2.INTER_AREA
    ):
        # https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def gui_video(self, width, fps):
        win_name = "Video"
        for f in self.video:
            f = self.ResizeWithAspectRatio(f, width=width)
            cv2.imshow(win_name, f)
            time.sleep(1 / fps)
            # cv2.waitKey(0) for pause
            if cv2.waitKey(5) & 0xFF == 27:  # ESC for exit
                break

    def gui_video_single_window(self, np_video, width, fps):
        win_name = "Video"
        for f in np_video:
            if width>0:
                f = self.ResizeWithAspectRatio(f, width=width)
            cv2.imshow(win_name, f)
            time.sleep(1 / fps)
            # cv2.waitKey(0) for pause
            if cv2.waitKey(5) & 0xFF == 27:  # ESC for exit
                break

    # TODO https://github.com/saman202/View_image_video_matrix
    # BUG Если нет мест в размере монитора для окна, то пропускать 
    def gui_video_multi_window_by_list_arr_video(self, list_arr_video, width, fps):   
        width_monitor = get_monitors()[0].width
        height_monitor = get_monitors()[0].height
 
        max_frame_count = 0
        for n in list_arr_video:

            if n.shape[0] > max_frame_count:
                max_frame_count = n.shape[0]
        window_titles = ["win" + str(i) for i in range(len(list_arr_video))]
        flag = True
        for i_frame in range(max_frame_count):
            w = 0
            h = 0
            for i, n in enumerate(window_titles):
                video = list_arr_video[i]
                cv2.namedWindow(n) 
                cv2.moveWindow(n, w, h)
                if len(video)>i_frame:
                    if width!=0:
                        image =  self.ResizeWithAspectRatio(video[i_frame], width=width)
                    else:
                        image = video[i_frame]
                    cv2.imshow(n, image)
                w = w + image.shape[0]
                if w > width_monitor-250:
                    w = 0
                    h = h + image.shape[1]
          
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if fps==0:
                cv2.waitKey(0)
            else:
                time.sleep(1 / fps)
        cv2.destroyAllWindows()