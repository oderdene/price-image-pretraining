import os

class Dataset:
    def __init__(self,):
        print("dataset instance is created.")
        pass
    def load(self,):
        print("dataset is loaded")
        pass
    def next_batch(self, batch_size):
        print("next batch request is called.")
        return [i for i in range(batch_size)]


ds = Dataset()
ds.load()

batch = ds.next_batch(batch_size=5)
print(batch)

