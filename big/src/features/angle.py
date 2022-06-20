import numpy as np


class Trans:
    def __init__(self):
        pass

    @staticmethod
    def mirror_xyz(list_xyz):
        ret = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
        print("mirror_xyzp return = ", ret)
        return ret

    @staticmethod
    def hand1orientation(list_xyz, view="palm"):
        ret = [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
        print("hand1orientation(view =", view, "return = ", ret)
        return ret
        

class Angle(Trans):
    def find_angle_xyz(self, a, b, b1, c):
        ba = a - b
        bc = c - b1
        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        angle = angle / 3.141592
        return round(angle, 4)

    def get_dict_handpoints(self):
        hand_points = {
            "wrist": [1, 0, 0, 9],
            "thumb_cmc": [0, 1, 1, 2],
            "thumb_mcp": [1, 2, 2, 3],
            "thumb_ip": [2, 3, 3, 4],
            "index_finger_mcp": [0, 5, 5, 6],
            "index_finger_pip": [5, 6, 6, 7],
            "index_finger_dip": [6, 7, 7, 8],
            "index_middle_mcp": [6, 5, 9, 10],
            "middle_finger_mcp": [0, 9, 9, 10],
            "middle_finger_pip": [9, 10, 10, 11],
            "middle_finger_dip": [10, 11, 11, 12],
            "middle_ring_mcp": [10, 9, 13, 14],
            "ring_mcp": [0, 13, 13, 14],
            "ring_finger_pip": [13, 14, 14, 15],
            "ring_finger_dip": [14, 15, 15, 16],
            "ring_rinky_mcp": [14, 13, 17, 18],
            "rinky_mcp": [0, 17, 17, 18],
            "rinky_finger_pip": [17, 18, 18, 19],
            "rinky_finger_dip": [18, 19, 19, 20],
        }
        return hand_points        

    def get_list_handkeys(self):
        return list(self.get_dict_handpoints().keys())   

    def xyz2angle_hand(self, xyz):
        hand_points = self.get_dict_handpoints()
        # https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
        angles = []
        sels = [n for n in hand_points.values()]
        # print("sels", sels)

        for n in sels:
            a, b, b1, c = (
                [xyz[n[0]][0], xyz[n[0]][1], xyz[n[0]][2]],
                [xyz[n[1]][0], xyz[n[1]][1], xyz[n[1]][2]],
                [xyz[n[2]][0], xyz[n[2]][1], xyz[n[2]][2]],
                [xyz[n[3]][0], xyz[n[3]][1], xyz[n[3]][2]],
            )
            # print("a, b, b1, c", a, b, b1, c)
            a = np.array(a)
            b = np.array(b)
            b1 = np.array(b1)
            c = np.array(c)
            angles.append(self.find_angle_xyz(a, b, b1, c))

        self.angle_hand = np.array(angles)
        return self.angle_hand

    def xyz2angles_hand(self, xyzs):
        list_xyzs = []
        for n in xyzs:
            list_xyzs.append(self.xyz2angle_hand(n))
        self.np_xyz = np.array(list_xyzs)
        return self.np_xyz