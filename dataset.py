import os
import random
import tensorflow.compat.v1 as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# https://github.com/mwdhont/SimCLRv1-keras-tensorflow/blob/master/SimCLR_data_util.py#L161

CROP_PROPRTION = 0.95

def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
            tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
            lambda: func(x),
            lambda: x)

def _compute_crop_shape(
        image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.
    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.
    Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)
    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(tf.rint(
            crop_proportion / aspect_ratio * image_width_float), tf.int32)
        crop_width = tf.cast(tf.rint(
            crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width
    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
                tf.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(tf.rint(
            crop_proportion * aspect_ratio *
            image_height_float), tf.int32)
        return crop_height, crop_width
    return tf.cond(
            aspect_ratio > image_width_float / image_height_float,
            _requested_aspect_ratio_wider_than_image,
            _image_wider_than_requested_aspect_ratio)

def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.
    Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
            image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
            image, offset_height, offset_width, crop_height, crop_width)
    image = tf.image.resize_bicubic([image], [height, width])[0]
    return image

def distorted_bounding_box_crop(image,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=100,
        scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
    Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                shape,
                bounding_boxes=bbox,
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        image = tf.image.crop_to_bounding_box(
                image, offset_y, offset_x, target_height, target_width)
        return image

def crop_and_resize(image, height, width):
    """Make a random crop and resize it to height `height` and width `width`.
    Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.
    Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
    """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    aspect_ratio = width / height
    image = distorted_bounding_box_crop(
            image,
            bbox,
            min_object_covered=0.7,
            aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
            area_range=(0.7, 1.0),
            max_attempts=100,
            scope=None)
    return tf.image.resize_bicubic([image], [height, width])[0]

def random_crop_with_resize(image, height, width, p=1.0):
    """Randomly crop and resize an image.
    Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: Probability of applying this transformation.
    Returns:
    A preprocessed image `Tensor`.
    """
    def _transform(image):  # pylint: disable=missing-docstring
        image = crop_and_resize(image, height, width)
        return image
    return random_apply(_transform, p=p, x=image)

def preprocess_image(image, height, width):
    image = random_crop_with_resize(image, height, width)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


class Dataset:
    def __init__(self, folder_path):
        self.height = 128 #256
        self.width  = 128 #256
        print("dataset is loading please wait...")
        self.image_paths = []
        for root, dirs, files in os.walk(folder_path):
            path = root.split(os.sep)
            for f in files:
                current_folder = os.path.join(*path)
                file_path      = os.path.join(current_folder, f)
                if file_path.endswith('.png')==True:
                    self.image_paths.append(file_path)
        print("dataset is loaded.")
        pass
    def next_batch(self, batch_size):
        batch_a       = []
        batch_b       = []
        random_images = [random.choice(self.image_paths) for _ in range(batch_size)]
        for i, row in enumerate(random_images):
            img       = cv.cvtColor(cv.imread(row), cv.COLOR_BGR2RGB)
            img       = tf.convert_to_tensor(np.asarray((img / 255)).astype("float32"))
            img       = tf.image.resize_bicubic([img], [self.height, self.width])[0] # optional
            augmented_a = random_crop_with_resize(img, self.height, self.width)
            augmented_a = tf.reshape(augmented_a, [self.height, self.width, 3])
            augmented_a = tf.clip_by_value(augmented_a, 0., 1.)
            batch_a.append(augmented_a)
            augmented_b = random_crop_with_resize(img, self.height, self.width)
            augmented_b = tf.reshape(augmented_b, [self.height, self.width, 3])
            augmented_b = tf.clip_by_value(augmented_b, 0., 1.)
            batch_b.append(augmented_b)
        return batch_a, batch_b


if __name__=="__main__":
    batch_size = 5

    f, axarr = plt.subplots(batch_size,2)

    ds = Dataset(folder_path="./dataset")
    batch_a, batch_b = ds.next_batch(batch_size=batch_size)

    for i in range(batch_size):
        axarr[i,0].imshow(batch_a[i])
        axarr[i,1].imshow(batch_b[i])

    plt.tight_layout()
    plt.show()

