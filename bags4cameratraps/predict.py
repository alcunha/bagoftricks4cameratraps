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

r"""Tool to evaluate classifiers.
Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""

import os
import random

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

from dataloader import CategoryMap, JsonInputProcessor
from bags import BalancedGroupSoftmax
from eval_utils import get_cls_bin_lists
from models import model_builder
from ssb import SquareRootSamplingBranch
from utils import log_flags

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='resnet50',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during evaluation.'))

flags.DEFINE_bool(
    'use_bags', default=False,
    help=('Use Balanced Group Softmax to train model'))

flags.DEFINE_integer(
    'empty_class_id', default=0,
    help=('Empty class id for balanced group softmax'))

flags.DEFINE_string(
    'model_weights', default=None,
    help=('Path to h5 file or ckp files to be loaded into the model.'))

flags.DEFINE_string(
    'sqrt_model_weights', default=None,
    help=('Path to h5 file or ckp files to the model trained using square-root'
          ' resampling. If provided, the prediction will use the main model for'
          ' bin4 prediction and the square-root model for the other bins.'))

flags.DEFINE_string(
    'train_json', default=None,
    help=('Path to json file containing the training annotations json on COCO'
          ' format'))

flags.DEFINE_string(
    'test_json', default=None,
    help=('Path to json file containing the test annotations json on COCO'
          ' format'))

flags.DEFINE_string(
    'dataset_dir', default=None,
    help=('Path to directory containing dataset images.'))

flags.DEFINE_string(
    'megadetector_results_json', default=None,
    help=('Path to json file containing megadetector results.'))

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))


if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('train_json')
flags.mark_flag_as_required('test_json')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('model_weights')


def build_input_data(data_json, category_map):
  input_data = JsonInputProcessor(data_json,
                    FLAGS.dataset_dir,
                    FLAGS.batch_size,
                    category_map,
                    megadetector_results_json=FLAGS.megadetector_results_json,
                    is_training=False,
                    output_size=FLAGS.input_size,
                    provide_filename=True,
                    batch_drop_remainder=False)

  dataset, _, _ = input_data.make_source_dataset()

  return dataset

def _decode_one_hot(one_hot_tensor):
  return tf.argmax(one_hot_tensor, axis=1).numpy()

def predict_classifier(model, dataset):
  file_names = []
  labels = []
  predictions = []
  count = 0
  confidences = []

  for batch, metadata in dataset:
    prediction = model(batch, training=False)
    label, file_name = metadata
    labels += list(_decode_one_hot(label))
    file_names += list(file_name.numpy())
    if FLAGS.sqrt_model_weights is not None:
        confidences += list(prediction.numpy())
    else:
        confidences += list(tf.nn.softmax(prediction, axis=1).numpy())
    predictions += list(_decode_one_hot(prediction))

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return labels, predictions, confidences, file_names


def load_model(num_classes, bags_header):
  model = model_builder.create(model_name=FLAGS.model_name,
                               num_classes=num_classes,
                               input_size=FLAGS.input_size,
                               bags_header=bags_header)

  model.load_weights(FLAGS.model_weights)

  if bags_header is not None:
    model = bags_header.create_prediction_model(model)

  return model

def load_ssb_model(num_classes, ssb):
  inst_model, base_model = model_builder.create(model_name=FLAGS.model_name,
                                                num_classes=num_classes,
                                                input_size=FLAGS.input_size,
                                                return_base_model=True,
                                                base_model_weights=None)
  inst_model.load_weights(FLAGS.model_weights)

  sqrt_model = model_builder.create(model_name=FLAGS.model_name,
                                    num_classes=num_classes,
                                    input_size=FLAGS.input_size,
                                    base_model_weights=None)
  sqrt_model.load_weights(FLAGS.sqrt_model_weights)

  model = ssb.create_prediction_model(base_model, inst_model, sqrt_model)

  return model

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()
  log_flags()

  if FLAGS.model_name == 'swin-s':
    tf.compat.v1.logging.info('The used Swin-s imlementation does not support'
      ' deterministic execution. Falling back to non deterministic execution.')
    os.environ['TF_DETERMINISTIC_OPS'] = '0'

  category_map = CategoryMap(FLAGS.train_json)
  cls_bins = get_cls_bin_lists(FLAGS.train_json, category_map)
  dataset = build_input_data(FLAGS.test_json, category_map)

  if FLAGS.sqrt_model_weights is not None:
    if FLAGS.use_bags:
      raise RuntimeError('Square-root side branch prediction is not compatible'
                         ' with Balanced Group Softmax')
    ssb = SquareRootSamplingBranch(cls_bins,
                                   category_map.get_num_classes(),
                                   FLAGS.input_size)
    model = load_ssb_model(category_map.get_num_classes(), ssb)
  else:
    bags_header = BalancedGroupSoftmax(
      FLAGS.train_json,
      category_map,
      FLAGS.empty_class_id) if FLAGS.use_bags else None

    model = load_model(category_map.get_num_classes(), bags_header)

  model.summary()

  labels, predictions, confidences, file_names = predict_classifier(model, dataset)

  for label, pred, conf, filename in zip(labels, predictions, confidences, file_names):
    print(f'Pred for {filename} ({category_map.index_to_category(label)}):'
          f' {category_map.index_to_category(pred)} ({100*conf.max():.2f}%)')

if __name__ == '__main__':
  app.run(main)
