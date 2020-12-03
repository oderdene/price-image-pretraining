import sys
import os
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

DATASET_PATH  = str  (config["DEFAULT"]["DATASET_PATH" ])


image_paths = []
for root, dirs, files in os.walk(DATASET_PATH):
    path = root.split(os.sep)
    for f in files:
        current_folder = os.path.join(*path)
        file_path      = os.path.join(current_folder, f)
        if file_path.endswith('.png')==True:
            image_paths.append(file_path)
            print(file_path)