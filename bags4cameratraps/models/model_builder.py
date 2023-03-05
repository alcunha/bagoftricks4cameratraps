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

import collections
import functools

import tensorflow as tf
import tensorflow.keras as keras
from .swintransformer import SwinTransformer
from .effnetv2.effnetv2_model import get_model as effnetv2

ModelSpecs = collections.namedtuple("ModelSpecs", [
    'name', 'func', 'input_size', 'classes', 'activation'])

def get_default_specs():
  return ModelSpecs(
    name='efficientnet-b0',
    func=keras.applications.EfficientNetB0,
    input_size=224,
    classes=2,
    activation='linear')

resnet50_spec = get_default_specs()._replace(
  name='resnet50',
  func=keras.applications.ResNet50,
  input_size=224
)

mobilenetv3large_spec = get_default_specs()._replace(
  name='mobilenetv3large',
  func=keras.applications.MobileNetV3Large,
  input_size=224
)

efficientnetv2_b2_spec = get_default_specs()._replace(
  name='efficientnetv2-b2',
  func=functools.partial(effnetv2, 'efficientnetv2-b2'),
  input_size=260
)

swin_s_spec = get_default_specs()._replace(
  name='swin-s',
  func=functools.partial(SwinTransformer, 'swin_small_224'),
  input_size=224
)

MODELS_SPECS = {
  'resnet50': resnet50_spec,
  'mobilenetv3large': mobilenetv3large_spec,
  'efficientnetv2-b2': efficientnetv2_b2_spec,
  'swin-s': swin_s_spec,
}

def _get_keras_base_model(specs, training, weights='imagenet'):
  base_model = None

  if specs.name.startswith('efficientnetv2'):
    base_model = specs.func(
      include_top=False,
      training=training,
      weights=weights)
  elif specs.name.startswith('swin'):
    base_model = specs.func(
      include_top=False,
      pretrained=(weights=='imagenet' or weights))
  else:
    base_model = specs.func(
      input_shape=(specs.input_size, specs.input_size, 3),
      include_top=False,
      pooling='avg',
      weights=weights)

  return base_model

def _create_model_from_specs(specs,
                             base_model_weights,
                             freeze_base_model,
                             bags_header=None,
                             return_base_model=False):
  training = not freeze_base_model

  image_input = keras.Input(shape=(specs.input_size, specs.input_size, 3))
  base_model = _get_keras_base_model(specs,
                                     training,
                                     weights=base_model_weights)
  base_model.trainable = training

  x = base_model(image_input, training=training)
  if bags_header is not None:
    outputs = bags_header.create_classif_header(x)
  else:
    outputs = [keras.layers.Dense(specs.classes,
                                activation=specs.activation)(x)]
  model = keras.models.Model(inputs=[image_input], outputs=outputs)

  if return_base_model:
    return model, base_model

  return model

def create(model_name,
           num_classes,
           input_size=None,
           freeze_base_model=True,
           bags_header=None,
           base_model_weights='imagenet',
           classifier_activation="linear",
           return_base_model=False):

  if model_name not in MODELS_SPECS.keys():
    raise RuntimeError('Model %s not implemented' % model_name)

  specs = MODELS_SPECS[model_name]
  specs = specs._replace(
    classes=num_classes,
    activation=classifier_activation,)
  if input_size is not None:
    specs = specs._replace(input_size=input_size)

  return _create_model_from_specs(specs,
                                  base_model_weights,
                                  freeze_base_model,
                                  bags_header,
                                  return_base_model)
