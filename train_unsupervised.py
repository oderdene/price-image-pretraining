import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import Dataset


class SimCLR(tf.keras.Model):
    def __init__(self,):
        super(SimCLR, self).__init__()
        self.width  = 512
        self.height = 512
        self.resnet_layer = tf.keras.applications.ResNet50(
                include_top = False,
                weights     = None,
                input_shape = (self.height, self.width, 3)
                )
        self.resnet.trainable = True
        self.h_layer      = tf.keras.layers.GlobalAveragePooling2D()
        self.projection_1 = tf.keras.layers.Dense(265, activation='relu')
        self.projection_2 = tf.keras.layers.Dense(128, activation='relu')
        self.z_layer      = tf.keras.layers.Dense(64)
    def call(self, inputs):
        x = self.resnet_layer(inputs)
        x = self.h_layer(x)
        x = self.projection_1(x)
        x = self.projection_2(x)
        return self.z_layer


if __name__=="__main__":
    print("train unsupervised way")
    batch_size = 5
    f, axarr   = plt.subplots(batch_size,2)
    ds         = Dataset(folder_path="./dataset")
    batch_a, batch_b = ds.next_batch(batch_size=batch_size)
    for i in range(batch_size):
        axarr[i,0].imshow(batch_a[i])
        axarr[i,1].imshow(batch_b[i])
    plt.tight_layout()
    plt.show()
    pass
