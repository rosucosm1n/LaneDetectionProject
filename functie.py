import cv2
import numpy as np

def getLinePoints(m, trapez, h, w):
    m3 = cv2.getPerspectiveTransform(np.float32(np.array([(w - 1, 0), (0, 0), (0, h - 1), (w - 1, h - 1)])), trapez)
    m5 = cv2.warpPerspective(m, m3, (w, h))
    whitePoints = np.argwhere(m5 > 1)

    top_x = -1
    top_y = -1
    bottom_x = -1
    bottom_y = -1

    for i in whitePoints:
        y = i[0]
        x = i[1]

        if y > bottom_y or bottom_y == -1 or x < bottom_x:
            bottom_y = y
            bottom_x = x
        if y < top_y or top_y == -1:
            top_y = y
            top_x = x

    top = (top_x, top_y)
    bottom = (bottom_x, bottom_y)

    return (top, bottom)