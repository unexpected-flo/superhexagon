from pynput.keyboard import Listener, Key


class Keylogger:
    def __init__(self):
        self.pressed_key = None
        l = Listener(on_press=self.press_callback, on_release=self.release_callback)
        l.start()

    def press_callback(self, key):
        self.pressed_key = key
        if key == Key.esc:
            return False

    def release_callback(self, key):
        self.pressed_key = None


if __name__ == "__main__":
    logger = Keylogger()
    while logger.pressed_key is not Key.esc:
        print(logger.pressed_key)
