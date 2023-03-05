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

r"""Tool to save weights from classifier base model without top layer.

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

from models import model_builder

# os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='resnet50',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'num_classes', default=None,
    help=('Number of classes of the model.'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_string(
    'base_model_path', default=None,
    help=('Path to save the base model weights on'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('base_model_path')
flags.mark_flag_as_required('num_classes')

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  model, base_model = model_builder.create(model_name=FLAGS.model_name,
                            num_classes=FLAGS.num_classes,
                            input_size=FLAGS.input_size,
                            return_base_model=True)

  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
  model.load_weights(checkpoint_path)
  model(tf.zeros([1, FLAGS.input_size, FLAGS.input_size, 3]))
  base_model.save_weights(FLAGS.base_model_path)

if __name__ == '__main__':
  app.run(main)
