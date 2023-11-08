import random
import string
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_random_str(length=5):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))
