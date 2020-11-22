import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from dataset import Dataset


BATCH_SIZE    = 5
EPOCHS        = 1
DECAY_STEPS   = 1000
LEARNING_RATE = 0.1
SAVE_STEP     = 100


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
    def call(self, inputs, training=False):
        x = self.resnet_layer(inputs, training=training)
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


if __name__=="__main__":
    print("train unsupervised way")

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits = True,
            reduction   = tf.keras.losses.Reduction.SUM)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate = LEARNING_RATE,
            decay_steps           = DECAY_STEPS)
    optimizer     = tf.keras.optimizers.SGD(lr_decayed_fn)

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

    ds           = Dataset(folder_path="./dataset")

    for epoch in range(EPOCHS):
        total_steps = int(len(ds.image_paths)/BATCH_SIZE)
        for step in range(total_steps):
            xis, xjs = ds.next_batch(batch_size=BATCH_SIZE)
            xis = tf.convert_to_tensor(xis)
            xjs = tf.convert_to_tensor(xjs)
            loss = train_step(xis, xjs, simclr_model, optimizer, criterion, temperature=0.1)
            print("epoch {} step {} of {}, loss {}".format(
                epoch, step, total_steps-1, loss
                ))
            ckpt.step.assign_add(1)
            if int(ckpt.step)%SAVE_STEP==0:
                save_path = ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            pass
        pass
    pass
