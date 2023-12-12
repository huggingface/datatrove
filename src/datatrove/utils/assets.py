import os


try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

BASE_PATH = os.path.dirname(files(__package__))

ASSETS_PATH = os.path.join(BASE_PATH, "assets")
DOWNLOAD_PATH = os.path.join(BASE_PATH, "download")
