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
        self.resnet_layer.trainable = True
        self.h_layer      = tf.keras.layers.GlobalAveragePooling2D()
        self.projection_1 = tf.keras.layers.Dense(265, activation='relu')
        self.projection_2 = tf.keras.layers.Dense(128, activation='relu')
        self.z_layer      = tf.keras.layers.Dense(64)
    def call(self, inputs):
        x = self.resnet_layer(inputs)
        x = self.h_layer(x)
        x = self.projection_1(x)
        x = self.projection_2(x)
        return self.z_layer(x)


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v

def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis)
        zjs = model(xjs)
        pass
    pass


if __name__=="__main__":
    print("train unsupervised way")

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
            reduction   = tf.keras.losses.Reduction.SUM)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate = 0.1,
            decay_steps           = 1000)
    optimizer     = tf.keras.optimizers.SGD(lr_decayed_fn)

    simclr_model = SimCLR()
    ds           = Dataset(folder_path="./dataset")

    batch_size = 5
    epochs     = 1
    for epoch in range(epochs):
        total_steps = int(len(ds.image_paths)/batch_size)
        for step in range(total_steps):
            print("epoch {} step {} of {}".format(epoch, step, total_steps-1))
            xis, xjs = ds.next_batch(batch_size=batch_size)
            xis = tf.convert_to_tensor(xis)
            xjs = tf.convert_to_tensor(xjs)
            loss = train_step(xis, xjs, simclr_model, optimizer, criterion, temperature=0.1)
            pass
    pass
