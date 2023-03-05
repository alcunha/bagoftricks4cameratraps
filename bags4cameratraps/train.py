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

r"""Tool to train classifiers.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility

Please note that the used swin transformer implementation does not support the
TensorFlow deterministic execution
"""

import os
import random

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

from dataloader import CategoryMap, JsonInputProcessor, TFRecordInputProcessor
from lr_schedulers import CosineDecayWithLinearWarmUpSchedule
from bags import BalancedGroupSoftmax
from models import model_builder
from utils import log_flags

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='resnet50',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_string(
    'base_model_weights', default='imagenet',
    help=('Path to h5 weights file to be loaded into the base model during'
          ' model build procedure.'))

flags.DEFINE_bool(
    'freeze_base_model', default=False,
    help=('Whether the base model should be frozen or trainable.'))

flags.DEFINE_bool(
    'use_bags', default=False,
    help=('Use Balanced Group Softmax to train model'))

flags.DEFINE_integer(
    'empty_class_id', default=0,
    help=('Empty class id for balanced group softmax'))

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during training.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_enum(
    'optimizer', default='adamw',
    enum_values=['sgd', 'adamw'],
    help=('Optimzer used for training'))

flags.DEFINE_float(
    'lr', default=0.001,
    help=('Initial learning rate'))

flags.DEFINE_float(
    'momentum', default=0,
    help=('Momentum for SGD optimizer'))

flags.DEFINE_float(
    'weight_decay', default=0,
    help=('The weight decay used for optimizers with decoupled weight decay'
          ' such as AdamW'))

flags.DEFINE_float(
    'label_smoothing', default=0,
    help=('When 0, no smoothing occurs. When > 0, we apply Label Smoothing to'
          ' the labels during training using this value for parameter e.'))

flags.DEFINE_float(
    'mixup_alpha', default=0.0,
    help=('Float that controls the strength of Mixup regularization.'))

flags.DEFINE_float(
    'cutmix_alpha', default=0.0,
    help=('FLoat that controls the strenght of Cutmix regularization.'))

flags.DEFINE_integer(
    'epochs', default=10,
    help=('Number of epochs to training for'))

flags.DEFINE_bool(
    'use_scaled_lr', default=False,
    help=('Scale the initial learning rate by batch size'))

flags.DEFINE_bool(
    'use_cosine_decay', default=False,
    help=('Apply cosine decay during training'))

flags.DEFINE_float(
    'warmup_epochs', default=0.0,
    help=('Duration of warmp of learning rate in epochs. It can be a'
          ' fractionary value as long will be converted to steps.'))

flags.DEFINE_enum(
    'loss_fn', default='sce',
    enum_values=['sce', 'focal'],
    help=('Loss used for training. sce: Softmax Cross Entropy; focal: Focal'
          ' Loss'))

flags.DEFINE_float(
    'focal_gamma', default=2.0,
    help=('Parameter gamma for focal loss'))

flags.DEFINE_float(
    'cb_beta', default=None,
    help=('Parameter beta for class-balanced loss'))

flags.DEFINE_integer(
    'ra_num_layers', default=None,
    help=('Number of operations to be applied by Randaugment'))

flags.DEFINE_integer(
    'ra_magnitude', default=None,
    help=('Magnitude for operations on Randaugment.'))

flags.DEFINE_string(
    'train_json', default=None,
    help=('Path to json file containing the training annotations json on COCO'
          ' format'))

flags.DEFINE_string(
    'val_json', default=None,
    help=('Path to json file containing the validation data'))

flags.DEFINE_string(
    'dataset_dir', default=None,
    help=('Path to directory containing dataset images.'))

flags.DEFINE_string(
    'training_tfrecords', default=None,
    help=('A file pattern for TFRecord files used for training'))

flags.DEFINE_string(
    'val_tfrecords', default=None,
    help=('A file pattern for TFRecord files used for validation'))

flags.DEFINE_string(
    'megadetector_results_json', default=None,
    help=('Path to json file containing megadetector results.'))

flags.DEFINE_enum(
    'sampling_strategy', default='instance',
    enum_values=['instance', 'sqrt'],
    help=('Sampling strategy used on the training set. instance: each training'
          ' instance has equal probability of being selected; sqrt: square-root'
          ' sampling'))

flags.DEFINE_integer(
    'validation_freq', default=1,
    help=('Specifies how many training epochs to run before a new validation '
          'run is performed.'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('train_json')
flags.mark_flag_as_required('model_dir')

def get_model(num_classes, bags_header):
  model = model_builder.create(model_name=FLAGS.model_name,
                               num_classes=num_classes,
                               input_size=FLAGS.input_size,
                               base_model_weights=FLAGS.base_model_weights,
                               bags_header=bags_header,
                               freeze_base_model=FLAGS.freeze_base_model)

  return model

def build_input_data(data_json,
                     file_pattern,
                     category_map,
                     is_training,
                     sampling_strategy='instance',
                     ra_num_layers=None,
                     ra_magnitude=None,
                     bags_header=None,
                     mixup_alpha=0.0,
                     cutmix_alpha=0.0):

  if file_pattern is None:
    input_data = JsonInputProcessor(data_json,
                    FLAGS.dataset_dir,
                    FLAGS.batch_size,
                    category_map,
                    megadetector_results_json=FLAGS.megadetector_results_json,
                    is_training=is_training,
                    sampling_strategy=sampling_strategy,
                    output_size=FLAGS.input_size,
                    bags_header=bags_header,
                    ra_num_layers=ra_num_layers,
                    ra_magnitude=ra_magnitude,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha)
  else:
    input_data = TFRecordInputProcessor(data_json,
                                    file_pattern,
                                    FLAGS.batch_size,
                                    category_map,
                                    is_training=is_training,
                                    sampling_strategy=sampling_strategy,
                                    output_size=FLAGS.input_size,
                                    bags_header=bags_header,
                                    ra_num_layers=ra_num_layers,
                                    ra_magnitude=ra_magnitude,
                                    mixup_alpha=mixup_alpha,
                                    cutmix_alpha=cutmix_alpha)

  return input_data.make_source_dataset()

def generate_lr_schedule(initial_learning_rate, steps_per_epoch, warmup_steps):
  decay_steps = FLAGS.epochs*steps_per_epoch - warmup_steps
  if FLAGS.use_cosine_decay:
    alpha = 0.0
  else:
    alpha = 1.0

  return CosineDecayWithLinearWarmUpSchedule(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    warmup_steps=warmup_steps,
    alpha=alpha)

def get_optimizer(lr, wd):
  if FLAGS.optimizer == 'sgd':
    return keras.optimizers.SGD(learning_rate=lr, momentum=FLAGS.momentum)
  elif FLAGS.optimizer == 'adamw':
    return tfa.optimizers.AdamW(learning_rate=lr,
                                weight_decay=wd)
  else:
    raise RuntimeError('Optimizers %s not implemented' % FLAGS.optimizer)

def get_loss_fn():
  if FLAGS.loss_fn == 'focal':
    return tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,
                                               gamma=FLAGS.focal_gamma)
  else:
    # We use logits, bug BAGS includes softmax on the classfication header
    from_logits = not FLAGS.use_bags
    return keras.losses.CategoricalCrossentropy(
                    from_logits=from_logits,
                    label_smoothing=FLAGS.label_smoothing)

def get_class_balanced_weights(num_per_cls):
  num_cls = len(num_per_cls)
  num_per_cls = [num_per_cls[i] for i in range(num_cls)]

  effective_num = 1.0 - tf.pow(FLAGS.cb_beta, num_per_cls)
  weights = (1.0 - FLAGS.cb_beta) / effective_num
  weights = weights / tf.math.reduce_sum(weights) * num_cls

  return {i: weights[i].numpy() for i in range(num_cls)}

def train_model(model,
                dataset,
                train_size,
                instances_per_cls,
                val_dataset,
                strategy):

  if FLAGS.cb_beta is not None:
    class_weight = get_class_balanced_weights(instances_per_cls)
    tf.compat.v1.logging.info('Class weight for loss: %s', (class_weight))
  else:
    class_weight = None

  warmup_steps = int(FLAGS.warmup_epochs * (train_size // FLAGS.batch_size))
  steps_per_epoch = train_size // FLAGS.batch_size

  lr = (FLAGS.lr * FLAGS.batch_size / 256) if FLAGS.use_scaled_lr else FLAGS.lr
  wd = FLAGS.weight_decay
  if FLAGS.use_cosine_decay or warmup_steps > 0:
    lr = generate_lr_schedule(lr, steps_per_epoch, warmup_steps)
    wd = generate_lr_schedule(wd, steps_per_epoch, warmup_steps)

  summary_dir = os.path.join(FLAGS.model_dir, "summaries")
  summary_callback = keras.callbacks.TensorBoard(summary_dir)

  checkpoint_filepath = os.path.join(FLAGS.model_dir, "ckp")
  checkpoint_callback = keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      save_freq='epoch')

  callbacks = [summary_callback, checkpoint_callback]

  with strategy.scope():
    optimizer = get_optimizer(lr, wd)
    loss = get_loss_fn()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['categorical_accuracy'])

  return model.fit(
    dataset,
    epochs=FLAGS.epochs,
    callbacks=callbacks,
    validation_data=val_dataset,
    validation_freq=FLAGS.validation_freq,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight)

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()
  log_flags(FLAGS.model_dir)

  if FLAGS.model_name == 'swin-s':
    tf.compat.v1.logging.info('The used Swin-s imlementation does not support'
      ' deterministic execution. Falling back to non deterministic execution.')
    os.environ['TF_DETERMINISTIC_OPS'] = '0'

  if FLAGS.dataset_dir is None and FLAGS.training_tfrecords is None:
    raise RuntimeError('Please specify --dataset_dir or --training_tfrecords')

  if (FLAGS.training_tfrecords is not None
      and FLAGS.megadetector_results_json is not None):
    raise RuntimeError('Current TFRecords implementation does not support'
                       ' Megadetector results.')

  category_map = CategoryMap(FLAGS.train_json)
  bags_header = BalancedGroupSoftmax(
    FLAGS.train_json,
    category_map,
    FLAGS.empty_class_id) if FLAGS.use_bags else None
  dataset, num_instances, instances_per_cls = build_input_data(FLAGS.train_json,
                                      FLAGS.training_tfrecords,
                                      category_map,
                                      is_training=True,
                                      sampling_strategy=FLAGS.sampling_strategy,
                                      bags_header=bags_header,
                                      ra_num_layers=FLAGS.ra_num_layers,
                                      ra_magnitude=FLAGS.ra_magnitude,
                                      mixup_alpha=FLAGS.mixup_alpha,
                                      cutmix_alpha=FLAGS.cutmix_alpha)

  if FLAGS.val_json is not None:
    if FLAGS.dataset_dir is None and FLAGS.val_tfrecords is None:
      raise RuntimeError('Please specify --dataset_dir or --val_tfrecords')

    val_dataset, _, _ = build_input_data(FLAGS.val_json,
                                      FLAGS.val_tfrecords,
                                      category_map,
                                      bags_header=bags_header,
                                      is_training=False)
  else:
    val_dataset = None

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  with strategy.scope():
    model = get_model(category_map.get_num_classes(), bags_header)
  model.summary()

  train_model(model,
              dataset,
              num_instances,
              instances_per_cls,
              val_dataset,
              strategy)

if __name__ == '__main__':
  app.run(main)
