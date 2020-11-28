import os
import sys
import signal
import random
import threading
import time
import configparser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
from dataset import Dataset

if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    #tf.config.experimental.set_virtual_device_configuration(
    #    physical_devices[0],
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])


config = configparser.ConfigParser()
config.read("config.ini")

BATCH_SIZE    = int  (config["DEFAULT"]["BATCH_SIZE"   ])
EPOCHS        = int  (config["DEFAULT"]["EPOCHS"       ])
SAVE_STEPS    = int  (config["DEFAULT"]["SAVE_STEPS"   ])
DATASET_PATH  = str  (config["DEFAULT"]["DATASET_PATH" ])
DECAY_STEPS   = int  (config["DEFAULT"]["DECAY_STEPS"  ])
LEARNING_RATE = float(config["DEFAULT"]["LEARNING_RATE"])



class ConvolutionalLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape=None, output_features=None, name=None):
        super(ConvolutionalLayer, self).__init__(name=name)
        self.conv_1          = tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
        self.maxpooling_1    = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2)
        self.conv_2          = tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), activation='relu')
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.maxpooling_2    = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2)
        self.conv_3          = tf.keras.layers.Conv2D(
                128, kernel_size=(3, 3), activation='relu')
        self.maxpooling_3    = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2)
        self.averagepooling  = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer    = tf.keras.layers.Dense(output_features)
    def call(self, inputs, training=False):
        x = self.conv_1(inputs)
        x = self.maxpooling_1(x)
        x = self.conv_2(x)
        x = self.normalization_1(x)
        x = self.maxpooling_2(x)
        x = self.conv_3(x)
        x = self.maxpooling_3(x)
        x = self.averagepooling(x)
        return self.output_layer(x)

class SimCLR(tf.keras.Model):
    def __init__(self,):
        super(SimCLR, self).__init__()
        self.conv_layer   = ConvolutionalLayer(
                #input_shape=(256, 256, 3),
                input_shape=(128, 128, 1),
                output_features=128,
                name="convolutional_features")
        self.projection_1 = tf.keras.layers.Dense(256, activation='relu')
        self.projection_2 = tf.keras.layers.Dense(128, activation='relu')
        self.z_layer      = tf.keras.layers.Dense(64)
    def call(self, inputs, training=False):
        x = self.conv_layer(inputs, training=training)
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
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    return v


negative_mask = get_negative_mask(BATCH_SIZE)

@tf.function
def train_step(xis, xjs, model, optimizer, criterion, temperature):
    with tf.GradientTape() as tape:
        zis = model(xis, training=True)
        zjs = model(xjs, training=True)

        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        l_pos  = _dot_simililarity_dim1(zis, zjs)
        l_pos  = tf.reshape(l_pos, (BATCH_SIZE, 1))
        l_pos /= temperature

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg  = _dot_simililarity_dim2(positives, negatives)

            labels = tf.zeros(BATCH_SIZE, dtype=tf.int32)

            l_neg  = tf.boolean_mask(l_neg, negative_mask)
            l_neg  = tf.reshape(l_neg, (BATCH_SIZE, -1))
            l_neg /= temperature

            logits = tf.concat([l_pos, l_neg], axis=1)
            loss  += criterion(y_pred=logits, y_true=labels)

        loss = loss/(2*BATCH_SIZE)
        pass

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def signal_handler(signal, frame):
    print("script killed...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


if __name__=="__main__":
    print("##### Unsupervised training of price images #####")

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
            reduction   = tf.keras.losses.Reduction.SUM)
    #lr_decayed_fn = tf.keras.experimental.CosineDecay(
    #        initial_learning_rate = LEARNING_RATE,
    #        decay_steps           = DECAY_STEPS)
    #optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    simclr_model = SimCLR()

    ckpt = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=optimizer,
            net=simclr_model
            )
    ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    if ckpt_manager.latest_checkpoint:
        print("Latest weights restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing weights from scratch")

    ds = Dataset(folder_path=DATASET_PATH, mem_length=10000)
    ds.update_dataset(batch_size=1024)

    def dataset_updater():
        while True:
            ds.update_dataset(batch_size=256)

    thread = threading.Thread(target=dataset_updater)
    thread.daemon = True
    thread.start()
    
    total_images = 300000.0

    for epoch in range(EPOCHS):
        total_steps = int(total_images/BATCH_SIZE)
        for step in range(total_steps):
            xis, xjs = ds.next_batch(batch_size=BATCH_SIZE)
            xis  = tf.convert_to_tensor(xis, dtype=tf.float32)
            xjs  = tf.convert_to_tensor(xjs, dtype=tf.float32)
            loss = train_step(xis, xjs, simclr_model, optimizer, criterion, temperature=0.5)
            print("cache {} epoch {} step {} of {}, loss {}".format(
                len(ds.cache), epoch, step, total_steps-1, loss
                ))
            ckpt.step.assign_add(1)
            if int(ckpt.step)%SAVE_STEPS==0:
                save_path = ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            pass
        pass
    pass
