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

import tensorflow as tf

class SquareRootSamplingBranch:
  def __init__(self, cls_bin_lists, num_classes, input_size):
    self.cls_bin_lists = cls_bin_lists
    self.num_classes = num_classes
    self.input_size = input_size
    self._create_masks()

  def _create_masks(self):
    inst_mask = self.cls_bin_lists[4]
    inst_mask = tf.one_hot(inst_mask, self.num_classes, dtype=tf.float32)
    inst_mask = tf.reduce_max(inst_mask, axis=0)
    inst_mask = tf.reshape(inst_mask, shape=(1,self.num_classes))

    sqrt_mask = tf.ones(tf.shape(inst_mask))
    sqrt_mask = sqrt_mask - inst_mask

    self.inst_mask = inst_mask
    self.sqrt_mask = sqrt_mask

  def create_prediction_model(self, base_model, instance_model, sqrt_model):
    image_input = tf.keras.Input(shape=(self.input_size, self.input_size, 3))
    x = base_model(image_input, training=False)

    inst_head_layer = tf.keras.layers.Dense(self.num_classes,
                                            activation='softmax')
    inst_head = inst_head_layer(x)
    inst_head_layer.set_weights(instance_model.layers[-1].get_weights())

    sqrt_head_layer = tf.keras.layers.Dense(self.num_classes,
                                            activation='softmax')
    sqrt_head = sqrt_head_layer(x)
    sqrt_head_layer.set_weights(sqrt_model.layers[-1].get_weights())

    inst_head = tf.keras.layers.Multiply()([inst_head, self.inst_mask])
    sqrt_head = tf.keras.layers.Multiply()([sqrt_head, self.sqrt_mask])
    preds = tf.keras.layers.Add()([inst_head, sqrt_head])

    model = tf.keras.models.Model(inputs=[image_input], outputs=[preds])

    return model
