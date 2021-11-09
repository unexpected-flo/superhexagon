from pynput.keyboard import Key, Controller
from random import choice, randint
from keyboard_logger import Keylogger
from time import sleep


def random_agent_active():
    print("Waiting to start, press enter")
    while logger.pressed_key is not Key.enter:
        pass
    print("Random agent active")
    while logger.pressed_key is not Key.esc:
        move = choice(keys)
        if move:
            keyboard.press(move)
        sleep(randint(1, 500)/1000)
        if move:
            keyboard.release(move)
    print("Random agent stopped")


if __name__ == '__main__':
    keys = [Key.left, Key.right, None]
    keyboard = Controller()
    logger = Keylogger()

    while logger.pressed_key is not Key.enter:
        pass
    sleep(0.5)

    random_agent_active()
