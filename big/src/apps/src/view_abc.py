import numpy as np


class View:
    def __init__(self) -> None:
        self.list_line_point_hand = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [7, 8],
            [9, 10],
            [10, 11],
            [11, 12],
            [13, 14],
            [14, 15],
            [15, 16],
            [17, 18],
            [18, 19],
            [19, 20],
            [0, 5],
            [0, 17],
            [5, 9],
            [9, 13],
            [13, 17]
        ]

    def cui(self, xyz: np.ndarray) -> None:
        """
        character User Interface
        """
        print("xyz", xyz)
