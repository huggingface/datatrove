import importlib.resources
import os


BASE_PATH = os.path.dirname(importlib.resources.files(__package__))

ASSETS_PATH = os.path.join(BASE_PATH, "assets")
DOWNLOAD_PATH = os.path.join(BASE_PATH, "download")
