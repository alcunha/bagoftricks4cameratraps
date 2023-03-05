# Copyright 2021 Fagner Cunha
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import flags

import tensorflow as tf
import tensorflow.keras as keras

import randaugment

flags.DEFINE_enum(
    'input_scale_mode', default='float32',
    enum_values=['tf', 'torch', 'caffe', 'uint8', 'float32'],
    help=('Mode for scaling input: tf scales image between -1 and 1;'
          ' torch uses image on scale 0-1 and normalizes inputs using ImageNet'
          ' mean and std; caffe convert the images from RGB to BGR, zero-center'
          ' each color channel with respect to the ImageNet dataset, without'
          ' scaling; uint8 uses image on scale 0-255; float32 uses image on'
          ' scale 0-1'))

FLAGS = flags.FLAGS

def random_crop(image,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.65, 1],
                min_object_covered=0):

  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=tf.zeros([0, 0, 4], tf.float32),
      min_object_covered=min_object_covered,
      area_range=area_range,
      aspect_ratio_range=aspect_ratio_range,
      use_image_if_no_bounding_boxes=True,
      max_attempts=100)

  image = tf.slice(image, bbox_begin, bbox_size)

  return image

def square_crop(image, bbox, min_size=112):
  img_height, img_width, _ = tf.split(tf.shape(image), num_or_size_splits=3)
  xmin, ymin, xmax, ymax = tf.split(bbox[0][0], num_or_size_splits=4)
  bbox_width = xmax - xmin
  bbox_height = ymax - ymin

  def castMultiply(x, y):
    y = tf.cast(y, dtype=tf.float32)
    return tf.cast(x * y, dtype=tf.int32)

  offset_width = castMultiply(xmin, img_width)
  offset_height = castMultiply(ymin, img_height)
  target_height = castMultiply(bbox_height, img_height)
  target_width = castMultiply(bbox_width, img_width)

  crop_size = tf.maximum(target_height, target_width)
  crop_size = tf.maximum(crop_size, min_size)
  crop_size = tf.minimum(tf.minimum(crop_size, img_height), img_width)

  center_x = offset_width + target_width//2
  center_y = offset_height + target_height//2

  offset_width_new = tf.maximum(0, center_x - crop_size//2)
  offset_height_new = tf.maximum(0, center_y - crop_size//2)

  offset_width_new = tf.cond(offset_width_new + crop_size > img_width,
                             lambda: img_width - crop_size,
                             lambda: offset_width_new)
  offset_height_new = tf.cond(offset_height_new + crop_size > img_height,
                              lambda: img_height - crop_size,
                              lambda: offset_height_new)

  image = tf.image.crop_to_bounding_box(image,
                                        offset_height_new[0],
                                        offset_width_new[0],
                                        crop_size[0],
                                        crop_size[0])

  return image

def scale_input(image):
  if FLAGS.input_scale_mode == 'float32':
    return tf.image.convert_image_dtype(image, dtype=tf.float32)
  elif FLAGS.input_scale_mode == 'uint8':
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)
  else:
    tf.compat.v1.logging.info('Using %s mode' % (FLAGS.input_scale_mode))
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return keras.applications.imagenet_utils.preprocess_input(
                  tf.cast(image, tf.float32), mode=FLAGS.input_scale_mode)

def preprocess_for_train(image,
                         image_size,
                         bbox=None,
                         use_square_crop=False,
                         ra_num_layers=None,
                         ra_magnitude=None):

  if use_square_crop and bbox is not None:
    image = square_crop(image, bbox)

  image = random_crop(image)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, size=(image_size, image_size))
  image = tf.image.random_flip_left_right(image)

  if ra_num_layers is not None and ra_magnitude is not None:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = randaugment.distort_image_with_randaugment(image,
                                                       ra_num_layers,
                                                       ra_magnitude)

  return image

def preprocess_for_eval(image, image_size, bbox=None, use_square_crop=False):
  if use_square_crop and bbox is not None:
    image = square_crop(image, bbox)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, size=(image_size, image_size))

  return image

def preprocess_image(image,
                     image_size,
                     is_training,
                     bboxes=None,
                     use_square_crop=False,
                     ra_num_layers=None,
                     ra_magnitude=None):
  if is_training:
    image = preprocess_for_train(image, image_size, bboxes, use_square_crop,
                                 ra_num_layers, ra_magnitude)
  else:
    image = preprocess_for_eval(image, image_size, bboxes, use_square_crop)

  image = scale_input(image)

  return image
