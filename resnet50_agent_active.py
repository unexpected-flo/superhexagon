import tensorflow as tf
import pathlib as P
from grab_screen import get_game_window_location
import mss
import numpy as np
import cv2
from pynput import keyboard


save_path = P.Path("./resnet50_trained_sftmx_BW/")
rn50_agent = tf.keras.models.load_model(save_path)

window = get_game_window_location()
offset_border = 8  # offsets found empirically to remove space around the game image
offset_top = 31
window = (window[0] + offset_border, window[1] + offset_top, window[2] - offset_border, window[3] - offset_border)

with mss.mss() as sct:
    pressed = None
    while True:
        sct_img = np.asarray(sct.grab(window))
        sct_img = np.flip(sct_img[:, :, :3], 2)
        sct_img = cv2.resize(sct_img, (224,224))
        # sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2RGB)

        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
        (T_img, sct_img) = cv2.threshold(sct_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # TODO thresholding is not good, going fullblack too often
        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_GRAY2RGB)

        cv2.imshow("screen", sct_img)
        key = cv2.waitKey(1)

        sct_img = sct_img.reshape(1, 224, 224, 3)
        pred = rn50_agent.predict(sct_img)
        choices = ["None", "left", "right"]
        print(choices[np.argmax(pred)])

        keys = [None, keyboard.Key.left, keyboard.Key.right]
        kbrd = keyboard.Controller()
        chosen = keys[np.argmax(pred)]
        print(chosen)


        if pressed:
            kbrd.release(pressed)
            pressed = None
        if chosen:
            pressed = chosen
            kbrd.press(chosen)





