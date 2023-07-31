import os


BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ASSETS_PATH = os.path.join(BASE_PATH, "assets")
DOWNLOAD_PATH = os.path.join(BASE_PATH, "download")
