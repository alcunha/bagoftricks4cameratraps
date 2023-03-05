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

import json
import functools

from absl import flags
import tensorflow as tf
import numpy as np
import pandas as pd

import mixing as mx
import preprocessing

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'suffle_buffer_size', default=10000,
    help=('Size of the buffer used to shuffle tfrecords'))

flags.DEFINE_integer(
    'num_readers', default=64,
    help=('Number of readers of TFRecord files'))

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _parse_single_example(feature_description, category_map, example_proto):
  features = tf.io.parse_single_example(example_proto,
                                        feature_description)
  image = features['image/encoded']
  label = features['image/object/class/label']
  label = tf.sparse.to_dense(label)[0]

  def _get_idx_label(label):
    return category_map.category_to_index(label.numpy())
  label = tf.py_function(func=_get_idx_label, inp=[label], Tout=tf.int32)

  return image, label

def _preprocess_image_from_tfrecord(image_size,
                                    cutmix_alpha,
                                    num_classes,
                                    is_training,
                                    bags_header,
                                    ra_num_layers,
                                    ra_magnitude,
                                    image,
                                    label):
  image = tf.io.decode_jpeg(image)
  image = preprocessing.preprocess_image(image,
                                    image_size,
                                    is_training,
                                    None,
                                    False,
                                    ra_num_layers,
                                    ra_magnitude)

  onehot_label = tf.one_hot(label, num_classes)
  if bags_header is not None:
    onehot_label = bags_header.process_label(onehot_label)

  features = {'image': image}
  labels = {'label': onehot_label}

  if cutmix_alpha:
    features['cutmix_mask'] = mx.cutmix_mask(cutmix_alpha,
                                             image_size,
                                             image_size)

  return features, labels

def _decode_bboxes(bboxes):
  xmin = bboxes['bbox_x']
  ymin = bboxes['bbox_y']
  xmax = xmin + bboxes['bbox_width']
  ymax = ymin + bboxes['bbox_height']

  bbox = tf.stack([xmin, ymin, xmax, ymax], axis=0)
  bbox = tf.reshape(bbox, shape=[1, 1, 4])

  return bbox

def _load_and_preprocess_image(dataset_dir,
                               image_size,
                               cutmix_alpha,
                               num_classes,
                               is_training,
                               bags_header,
                               use_square_crop,
                               ra_num_layers,
                               ra_magnitude,
                               provide_filename,
                               filename,
                               label,
                               bboxes):
  bbox = _decode_bboxes(bboxes)
  image = tf.io.read_file(dataset_dir + filename)
  image = tf.io.decode_jpeg(image, channels=3)
  image = preprocessing.preprocess_image(image,
                                    image_size,
                                    is_training,
                                    bbox,
                                    use_square_crop,
                                    ra_num_layers,
                                    ra_magnitude)

  onehot_label = tf.one_hot(label, num_classes)
  if bags_header is not None:
    onehot_label = bags_header.process_label(onehot_label)

  features = {'image': image}
  if provide_filename:
    labels = {'label': (onehot_label, filename)}
  else:
    labels = {'label': onehot_label}

  if cutmix_alpha:
    features['cutmix_mask'] = mx.cutmix_mask(cutmix_alpha,
                                             image_size,
                                             image_size)

  return features, labels

class JsonInputProcessor:
  def __init__(self,
               dataset_json,
               dataset_dir,
               batch_size,
               category_map,
               megadetector_results_json=None,
               conf_threshold=0.6,
               is_training=False,
               use_eval_preprocess=False,
               output_size=224,
               bags_header=None,
               ra_num_layers=None,
               ra_magnitude=None,
               mixup_alpha=0,
               cutmix_alpha=0,
               default_empty_label=0,
               sampling_strategy='instance',
               provide_filename=False,
               batch_drop_remainder=True):
    self.dataset_json = dataset_json
    self.dataset_dir = dataset_dir
    self.batch_size = batch_size
    self.category_map = category_map
    self.megadetector_results_json = megadetector_results_json
    self.conf_threshold = conf_threshold
    self.is_training = is_training
    self.output_size = output_size
    self.bags_header = bags_header
    self.ra_num_layers = ra_num_layers
    self.ra_magnitude = ra_magnitude
    self.mixup_alpha = mixup_alpha
    self.cutmix_alpha = cutmix_alpha
    self.default_empty_label = default_empty_label
    self.sampling_strategy = sampling_strategy
    self.batch_drop_remainder = batch_drop_remainder
    self.provide_filename = provide_filename
    self.preprocess_for_train = is_training and not use_eval_preprocess
    self.num_instances = 0
    self.instances_per_cls = {}

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])
    if 'annotations' in json_data.keys():
      annotations = pd.DataFrame(json_data['annotations'])
      images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')
    else:
      images['category_id'] = self.default_empty_label

    images['category_id'] = images['category_id'].apply(
                                          self.category_map.category_to_index)

    if self.megadetector_results_json is not None:
      with tf.io.gfile.GFile(self.megadetector_results_json, 'r') as json_file:
        json_data = json.load(json_file)
      megadetector_preds = pd.DataFrame(json_data['images'])
      images = pd.merge(images,
                      megadetector_preds,
                      how='left',
                      on='file_name')
    else:
      images['detections'] = np.nan

    return images

  def _resample_dataset(self, dataset, sampling_factor=1/2):
    tf.compat.v1.logging.info('Using sampling with q=%f' % (sampling_factor))

    instances = [self.instances_per_cls[i] for i in range(self.num_classes)]
    instances = tf.convert_to_tensor(instances, dtype=tf.float32)
    initial_dist = instances / tf.reduce_sum(instances)
    target_dist = tf.pow(instances, sampling_factor)
    target_dist = target_dist / tf.reduce_sum(target_dist)

    def class_func(file_name, category_id, bboxes):
      return category_id

    resampler = tf.data.experimental.rejection_resample(
      class_func, target_dist=target_dist, initial_dist=initial_dist)

    dataset = dataset.apply(resampler).map(
        lambda extra_label, features_and_label: features_and_label)

    return dataset

  def _prepare_bboxes(self, metadata):
    def _get_first_bbox(row):
      bbox = row['detections']
      if len(bbox) > 0 and bbox[0]['conf'] > self.conf_threshold:
        bbox = bbox[0]['bbox']
      else:
        bbox = [0.0, 0.0, 1.0, 1.0]
      return bbox

    metadata['detections'] = metadata['detections'].apply(
                            lambda d: d if isinstance(d, list) else [])
    metadata['bbox'] = metadata.apply(_get_first_bbox, axis=1)
    bboxes = pd.DataFrame(metadata.bbox.tolist(),
                    columns=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])

    return bboxes.to_dict('list')

  def make_source_dataset(self):
    metadata = self._load_metadata()
    self.num_instances = len(metadata)
    self.num_classes = self.category_map.get_num_classes()
    self.instances_per_cls = dict(metadata['category_id'].value_counts())

    bboxes = self._prepare_bboxes(metadata)

    dataset = tf.data.Dataset.from_tensor_slices((
        metadata.file_name,
        metadata.category_id,
        bboxes))

    if self.is_training:
      dataset = dataset.shuffle(self.num_instances,
                                reshuffle_each_iteration=True).repeat()
      if self.sampling_strategy == 'sqrt':
        dataset = self._resample_dataset(dataset, sampling_factor=1/2)

    use_square_crop = self.megadetector_results_json is not None
    dataset = dataset.map(
        functools.partial(_load_and_preprocess_image, self.dataset_dir,
                          self.output_size, self.cutmix_alpha, self.num_classes,
                          self.preprocess_for_train, self.bags_header,
                          use_square_crop, self.ra_num_layers,
                          self.ra_magnitude, self.provide_filename),
        num_parallel_calls=AUTOTUNE).batch(
          self.batch_size, drop_remainder=self.batch_drop_remainder)
    dataset = dataset.map(
        functools.partial(mx.mixing, self.batch_size, self.mixup_alpha,
                          self.cutmix_alpha),
        num_parallel_calls=AUTOTUNE)

    if self.bags_header is not None:
      def _generate_masks(inputs, outputs):
        masks = self.bags_header.generate_balancing_mask(outputs)
        return (inputs, outputs, masks)
      dataset = dataset.map(_generate_masks,
                            num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset, self.num_instances, self.instances_per_cls


class TFRecordInputProcessor:
  def __init__(self,
               dataset_json,
               file_pattern,
               batch_size,
               category_map,
               is_training=False,
               use_eval_preprocess=False,
               output_size=224,
               bags_header=None,
               ra_num_layers=None,
               ra_magnitude=None,
               mixup_alpha=0,
               cutmix_alpha=0,
               default_empty_label=0,
               sampling_strategy='instance',
               batch_drop_remainder=True):
    self.dataset_json = dataset_json
    self.file_pattern = file_pattern
    self.batch_size = batch_size
    self.category_map = category_map
    self.is_training = is_training
    self.output_size = output_size
    self.bags_header = bags_header
    self.ra_num_layers = ra_num_layers
    self.ra_magnitude = ra_magnitude
    self.mixup_alpha = mixup_alpha
    self.cutmix_alpha = cutmix_alpha
    self.default_empty_label = default_empty_label
    self.sampling_strategy = sampling_strategy
    self.batch_drop_remainder = batch_drop_remainder
    self.preprocess_for_train = is_training and not use_eval_preprocess
    self.num_instances = 0
    self.instances_per_cls = {}

    self.feature_description = {
    'image/height': tf.io.FixedLenFeature((), tf.int64, default_value=1),
    'image/width': tf.io.FixedLenFeature((), tf.int64, default_value=1),
    'image/filename':
        tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/source_id':
        tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/key/sha256':
        tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/format':
        tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])
    if 'annotations' in json_data.keys():
      annotations = pd.DataFrame(json_data['annotations'])
      images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')
    else:
      images['category_id'] = self.default_empty_label

    images['category_id'] = images['category_id'].apply(
                                          self.category_map.category_to_index)

    return images

  def _resample_dataset(self, dataset, sampling_factor=1/2):
    tf.compat.v1.logging.info('Using sampling with q=%f' % (sampling_factor))

    instances = [self.instances_per_cls[i] for i in range(self.num_classes)]
    instances = tf.convert_to_tensor(instances, dtype=tf.float32)
    initial_dist = instances / tf.reduce_sum(instances)
    target_dist = tf.pow(instances, sampling_factor)
    target_dist = target_dist / tf.reduce_sum(target_dist)

    def class_func(file_name, category_id):
      return category_id

    resampler = tf.data.experimental.rejection_resample(
      class_func, target_dist=target_dist, initial_dist=initial_dist)

    dataset = dataset.apply(resampler).map(
        lambda extra_label, features_and_label: features_and_label)

    return dataset

  def make_source_dataset(self):
    metadata = self._load_metadata()
    self.num_instances = len(metadata)
    self.num_classes = self.category_map.get_num_classes()
    self.instances_per_cls = dict(metadata['category_id'].value_counts())

    filenames = tf.io.gfile.glob(self.file_pattern)
    dataset_files = tf.data.Dataset.list_files(self.file_pattern,
                                               shuffle=self.is_training)

    num_readers = FLAGS.num_readers
    if num_readers > len(filenames):
      num_readers = len(filenames)
      tf.compat.v1.logging.info('num_readers has been reduced to %d to match'
                       ' input file shards.' % num_readers)
    dataset = dataset_files.interleave(
      lambda x: tf.data.TFRecordDataset(x,
                        buffer_size=8 * 1000 * 1000).prefetch(AUTOTUNE),
                        cycle_length=num_readers,
                        num_parallel_calls=AUTOTUNE)

    if self.is_training:
      dataset = dataset.shuffle(FLAGS.suffle_buffer_size,
                                reshuffle_each_iteration=True).repeat()

    dataset = dataset.map(
      functools.partial(_parse_single_example,
                        self.feature_description,
                        self.category_map),
      num_parallel_calls=AUTOTUNE)

    if self.is_training and self.sampling_strategy == 'sqrt':
      dataset = self._resample_dataset(dataset, sampling_factor=1/2)

    dataset = dataset.map(
      functools.partial(_preprocess_image_from_tfrecord, self.output_size,
                          self.cutmix_alpha, self.num_classes,
                          self.preprocess_for_train, self.bags_header,
                          self.ra_num_layers, self.ra_magnitude),
      num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE).batch(
          self.batch_size, drop_remainder=self.batch_drop_remainder)
    dataset = dataset.map(
        functools.partial(mx.mixing, self.batch_size, self.mixup_alpha,
                          self.cutmix_alpha),
        num_parallel_calls=AUTOTUNE)

    if self.bags_header is not None:
      def _generate_masks(inputs, outputs):
        masks = self.bags_header.generate_balancing_mask(outputs)
        return (inputs, outputs, masks)
      dataset = dataset.map(_generate_masks,
                            num_parallel_calls=AUTOTUNE)

    return dataset, self.num_instances, self.instances_per_cls

class CategoryMap:
  def __init__(self, dataset_json):
    with open(dataset_json) as json_file:
      data = json.load(json_file)

    category2idx = {}
    idx2category = {}
    category2name = {}
    category_list = []

    categories = pd.DataFrame(data['annotations'])['category_id'].unique()
    categories = sorted(categories)
    for idx, category in enumerate(categories):
      category2idx[category] = idx
      idx2category[idx] = category
      category_list.append(category)

    category2name = {cat['id']: cat['name'] for cat in data['categories']
                      if cat['id'] in category_list}

    self.category2idx = category2idx
    self.idx2category = idx2category
    self.category2name = category2name
    self.num_classes = len(self.category2idx)
    self.category_list = category_list

  def category_to_index(self, category):
    return self.category2idx[category]

  def index_to_category(self, index):
    return self.idx2category[index]

  def category_to_name(self, category):
    return self.category2name[category]

  def get_category_list(self):
    return self.category_list

  def get_num_classes(self):
    return self.num_classes
