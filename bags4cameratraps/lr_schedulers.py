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

import math

import tensorflow as tf
import tensorflow.keras as keras

def linear_warmup(initial_learning_rate, current_step, warmup_steps):
  return current_step * initial_learning_rate / warmup_steps


def cosine_decay(initial_learning_rate,
                 current_step,
                 decay_steps,
                 alpha):

  dtype = initial_learning_rate.dtype

  current_step = tf.minimum(current_step, decay_steps)
  cosine_decayed = 0.5 * (1.0 + tf.math.cos(
      tf.constant(math.pi, dtype=dtype) * current_step / decay_steps))
  decayed = (1 - alpha) * cosine_decayed + alpha

  return tf.multiply(initial_learning_rate, decayed)


class CosineDecayWithLinearWarmUpSchedule(keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self,
               initial_learning_rate,
               decay_steps,
               warmup_steps=0,
               alpha=0.0,
               name=None):
    super(CosineDecayWithLinearWarmUpSchedule, self).__init__()

    if decay_steps == 0:
      alpha = 1.0
      decay_steps = 1

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.warmup_steps = warmup_steps
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "CosineDecayWithLinearWarmUpSchedule"):
      initial_learning_rate = tf.convert_to_tensor(
      self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = tf.cast(self.decay_steps, dtype)
      warmup_steps = tf.cast(self.warmup_steps, dtype)
      alpha =  tf.cast(self.alpha, dtype)

      current_step = tf.cast(step, dtype)
      steps_after_warmup = current_step - warmup_steps
      return tf.cond(current_step < warmup_steps,
                      lambda: linear_warmup(initial_learning_rate,
                                            current_step,
                                            warmup_steps),
                      lambda: cosine_decay(initial_learning_rate,
                                          steps_after_warmup,
                                          decay_steps,
                                          alpha))

  def get_config(self):
    return {
      "initial_learning_rate": self.initial_learning_rate,
      "decay_steps": self.decay_steps,
      "warmup_steps": self.warmup_steps,
      "alpha": self.alpha,
      "name": self.name
    }
