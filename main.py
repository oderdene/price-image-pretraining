import os
import random
from dataset import *


ds = Dataset(folder_path="./dataset")

for _ in range(5):
    batch = ds.next_batch(batch_size=5)
    print(batch)

