import pathlib as P
import mss
import mss.tools
from pynput import keyboard
from grab_screen import get_game_window_location
from keyboard_logger import Keylogger


if __name__ == '__main__':
    destination_folder = P.Path("./train_data")
    window = get_game_window_location()
    # the window returned by windows functions is somehow a bit larger than the   real window.
    # Offset found empirically to trim the useless desktop parts around the window.
    offset_border = 8
    offset_top = 31
    window = (window[0] + offset_border, window[1] + offset_top, window[2] - offset_border, window[3] - offset_border)
    keylog = Keylogger()
    while keylog.pressed_key is not keyboard.Key.enter:
        pass
    with mss.mss() as sct:
        i = len(list(destination_folder.iterdir()))  # To avoid overwriting files that may already be there from previous recording
        i += 1
        while keylog.pressed_key is not keyboard.Key.esc:
            img = sct.grab(window)
            mss.tools.to_png(img.rgb, img.size, output="{}/{}_{}.png".format(destination_folder, i, keylog.pressed_key))
            i += 1
        img = sct.grab(window)
        mss.tools.to_png(img.rgb, img.size, output="{}/{}_END_CAPTURE.png".format(destination_folder, i))
