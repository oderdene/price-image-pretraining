import os
import random

class Dataset:
    def __init__(self, folder_path):
        print("dataset is loading please wait...")
        self.image_paths = []
        for root, dirs, files in os.walk(folder_path):
            path = root.split(os.sep)
            for f in files:
                current_folder = os.path.join(*path)
                file_path = os.path.join(current_folder, f)
                if file_path.endswith('.png')==True:
                    self.image_paths.append(file_path)
        print("dataset is loaded.")
        pass
    def load(self,):
        print("dataset is loaded")
        pass
    def next_batch(self, batch_size):
        return [random.choice(self.image_paths) for _ in range(batch_size)]


ds = Dataset(folder_path="./dataset")

for _ in range(5):
    batch = ds.next_batch(batch_size=5)
    print(batch)

