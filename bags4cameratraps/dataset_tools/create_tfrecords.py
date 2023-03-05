# Copyright 2022 Fagner Cunha
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

import contextlib2
import hashlib
import json
import os
import random

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

import dataset_util
from tf_record_creation_util import open_sharded_output_tfrecords

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'annotations_json', default=None,
    help=('Path to json file containing the annotations json on COCO format'))

flags.DEFINE_string(
    'dataset_dir', default=None,
    help=('Path to directory containing dataset images.'))

flags.DEFINE_string(
    'tfrecord_path', default=None,
    help=('Path to save tfrecords to.'))

flags.DEFINE_integer(
    'images_per_shard', default=800,
    help=('Number of images per shard'))

flags.DEFINE_bool(
    'shufle_images', default=True,
    help=('Shufle images before to write to tfrecords'))

flags.mark_flag_as_required('annotations_json')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('tfrecord_path')

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

def _get_image_dimensions_from_file(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  height = tf.shape(image)[0]
  width = tf.shape(image)[1]

  return height, width

def create_tf_example(image,
                      dataset_base_dir,
                      annotations):

  filename = image['file_name'].split('/')[-1]
  image_id = image['id']

  image_path = os.path.join(dataset_base_dir, image['file_name'])
  if not tf.io.gfile.exists(image_path):
    return None

  with tf.io.gfile.GFile(image_path, 'rb') as image_file:
    encoded_image_data = image_file.read()
  key = hashlib.sha256(encoded_image_data).hexdigest()

  height, width = _get_image_dimensions_from_file(image_path)
  classes = []
  for annotation in annotations:
    category_id = annotation['category_id']
    classes.append(category_id)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example

def create_tf_record_from_images_list(images,
                                      annotations_index,
                                      dataset_base_dir,
                                      output_path):

  num_shards = 1 + (len(images) // FLAGS.images_per_shard)
  total_image_skipped = 0

  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = open_sharded_output_tfrecords(
        tf_record_close_stack, output_path, num_shards)

    for index, image in enumerate(images):
      image_id = image['id']
      if image_id not in annotations_index:
        annotations_index[image_id] = []
      tf_example = create_tf_example(image,
                                     dataset_base_dir,
                                     annotations_index[image_id])

      if tf_example is not None:
        output_shard_index = index % num_shards
        output_tfrecords[output_shard_index].write(
            tf_example.SerializeToString())
      else:
        total_image_skipped += 1

    tf.compat.v1.logging.info('%d images not found.', total_image_skipped)

def _get_annotations_index(annotations):
  annotations_index = {}
  for annotation in annotations:
    image_id = annotation['image_id']
    if image_id not in annotations_index:
      annotations_index[image_id] = []
    annotations_index[image_id].append(annotation)

  return annotations_index

def _create_tf_record(annotations_json,
                      dataset_dir,
                      tfrecord_path):

  with tf.io.gfile.GFile(annotations_json, 'r') as json_file:
    json_data = json.load(json_file)
  images = json_data['images']
  if 'annotations' in json_data.keys():
    annot_index = _get_annotations_index(json_data['annotations'])
  else:
    annot_index = {}

  if FLAGS.shufle_images:
    random.shuffle(images)

  create_tf_record_from_images_list(images,
                                    annot_index,
                                    dataset_dir,
                                    tfrecord_path)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  _create_tf_record(FLAGS.annotations_json,
                    FLAGS.dataset_dir,
                    FLAGS.tfrecord_path)

if __name__ == '__main__':
  app.run(main)
