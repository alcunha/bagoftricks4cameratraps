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

import os

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

flags.DEFINE_integer(
        'resize_min_size',
        600,
        'Size which the shortest image side will be resized to.',
        lower_bound=0)
flags.DEFINE_integer(
        'resize_max_size',
        1024,
        'Size which the largest image side is allowed to be resized to.',
        lower_bound=0)
flags.DEFINE_string(
        'input_images_dir',
        None,
        'Location of images to be resized.')
flags.DEFINE_string(
        'output_images_dir',
        None,
        'Location to where the resized images will be writen.')

FLAGS = flags.FLAGS
AUTOTUNE = tf.data.experimental.AUTOTUNE

def _read_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  return image

def _write_image(image, image_path):
  image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  image = tf.image.encode_jpeg(image)
  fwrite = tf.io.write_file(image_path, image)

  return fwrite


def _resize_image(image):
  height = tf.cast(tf.shape(image)[0], tf.float32)
  width = tf.cast(tf.shape(image)[1], tf.float32)
  min_side = tf.constant(FLAGS.resize_min_size, dtype=tf.float32)
  max_side = tf.constant(FLAGS.resize_max_size, dtype=tf.float32)

  shortest_side = tf.minimum(height, width)
  largest_side = tf.maximum(height, width)
  scale = min_side / shortest_side

  scale = tf.cond(
      largest_side * scale > max_side,
      lambda: max_side / largest_side,
      lambda: scale
  )

  scaled_height = height * scale
  scaled_width = width * scale

  image = tf.image.resize(image, [scaled_height, scaled_width])

  return image

def _map_fn(image_path, output_image_path):
  image = _read_image(image_path)
  image = _resize_image(image)
  _write_image(image, output_image_path)

  return image_path, output_image_path

def _list_all_pic_from_dir(basedir,
                          white_list_formats = {'jpg', 'jpeg'}):
  """List all pictures from a directory recursively

  # Arguments
    basedir: directory base for the search
    white_list_formats: a list of allowed file extensions

  # Returns
    A list of all files

  """

  all_files = []

  for root, _, filenames in os.walk(basedir):
    for filename in filenames:
      fname = os.path.join(root,filename)

      if white_list_formats is not None:
        is_valid = False
        for extension in white_list_formats:
          if fname.lower().endswith('.' + extension):
            is_valid = True
            break
        if is_valid:
          all_files.append(fname)
      else:
        all_files.append(fname)

  return all_files

def _generate_output_path(file_name, input_images_dir, output_images_dir):
  fname = os.path.relpath(file_name, input_images_dir)

  return os.path.join(output_images_dir, fname)

def _generate_output_list(image_list, input_images_dir, output_images_dir):
  image_list = [ _generate_output_path(image_path,
                                       input_images_dir,
                                       output_images_dir)
                for image_path in image_list]

  return image_list

def resize_images(input_images_dir, output_images_dir):
  logging.info('reading image list')
  image_list = _list_all_pic_from_dir(input_images_dir)
  logging.info("%d images found" % len(image_list))

  output_image_list = _generate_output_list(image_list,
                                            input_images_dir,
                                            output_images_dir)

  dataset = tf.data.Dataset.from_tensor_slices((image_list, output_image_list))
  dataset = dataset.map(_map_fn, num_parallel_calls=AUTOTUNE)
  dataset = dataset.apply(tf.data.experimental.ignore_errors())

  for input_image, _ in dataset.take(len(image_list)):
    logging.info("resizing %s" % input_image.numpy())

def main(_):
  if not FLAGS.input_images_dir:
    raise RuntimeError('Must specify --input_images_dir to resize images.')

  if not FLAGS.output_images_dir:
    raise RuntimeError('Must specify --output_images_dir to resize images.')

  resize_images(FLAGS.input_images_dir, FLAGS.output_images_dir)

if __name__ == '__main__':
  app.run(main)
