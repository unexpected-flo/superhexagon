import pathlib as P
from grab_screen import get_game_window_location
import mss
import numpy as np
import cv2


def find_cursor_coords(image, resized=True):
    template = cv2.imread("./template_1.png")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = maxLoc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]

    x_ratio = 1
    y_ratio = 1
    if resized:
        # game resolution 480x768
        # input resolution 224x224
        # ratio horizontal 0.29 vertical 0.46
        x_ratio = 0.29
        y_ratio = 0.46

    return int((startX + (endX-startX)/2)*x_ratio), int((startY + (endY-startY)/2)*y_ratio)




if __name__ == '__main__':
    window = get_game_window_location()
    offset_border = 8  # offsets found empirically to remove space around the game image
    offset_top = 31
    window = (window[0] + offset_border, window[1] + offset_top, window[2] - offset_border, window[3] - offset_border)
    print(window)
    with mss.mss() as sct:
        while True:
            sct_img = np.asarray(sct.grab(window))
            x, y = find_cursor_coords(sct_img)
            sct_img = cv2.resize(sct_img, (224, 224))
            cv2.circle(sct_img, (x, y), 3, (57, 255, 20), -1)
            cv2.imshow("screen", sct_img)
            key = cv2.waitKey(1)
