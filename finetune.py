import os
import sys
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import Dataset

if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0], 
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])


BATCH_SIZE    = 64
EPOCHS        = 1
DECAY_STEPS   = 1000
LEARNING_RATE = 0.1
SAVE_STEP     = 100


class ConvolutionalLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape=None, output_features=None, name=None):
        super(ConvolutionalLayer, self).__init__(name=name)
        self.conv_1          = tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
        self.maxpooling_1    = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2)
        self.dropout_1       = tf.keras.layers.Dropout(0.2)
        self.conv_2          = tf.keras.layers.Conv2D(
                64, kernel_size=(3, 3), activation='relu')
        self.normalization_1 = tf.keras.layers.BatchNormalization()
        self.maxpooling_2    = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=2)
        self.averagepooling  = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout_2       = tf.keras.layers.Dropout(0.5)
        self.output_layer    = tf.keras.layers.Dense(output_features)
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.maxpooling_1(x)
        x = self.dropout_1(x)
        x = self.conv_2(x)
        x = self.normalization_1(x)
        x = self.maxpooling_2(x)
        x = self.averagepooling(x)
        x = self.dropout_2(x)
        return self.output_layer(x)

class SimCLR(tf.keras.Model):
    def __init__(self,):
        super(SimCLR, self).__init__()
        self.conv_layer   = ConvolutionalLayer(
                input_shape=(256, 256, 3),
                output_features=256,
                name="convolutional_features")
        self.projection_1 = tf.keras.layers.Dense(256, activation='relu')
        self.projection_2 = tf.keras.layers.Dense(128, activation='relu')
        self.z_layer      = tf.keras.layers.Dense(64)
    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        x = self.projection_1(x)
        x = self.projection_2(x)
        return self.z_layer(x)



if __name__=="__main__":
    print("##### Freeze layers and finetune for classification #####")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

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

    # neural surgical procedure :P
    for layer in simclr_model.layers:
        print(layer.name)

    conv_layer = simclr_model.get_layer('convolutional_features')
    sample_input  = tf.random.normal(shape=(5, 256, 256, 3))
    sample_output = conv_layer(sample_input)
    print(sample_output.shape)
    print(sample_output)

    conv_layer_weights = conv_layer.get_weights()
    print("convolutional layer weights :")
    print(conv_layer_weights)

