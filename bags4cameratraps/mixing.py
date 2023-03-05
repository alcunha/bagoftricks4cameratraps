# Copyright 2021 Fagner Cunha
# Copyright 2021 Google Research. All Rights Reserved.
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
#
# This file has been modified by Fagner Cunha to keep only the mixup and cutmix
# as independent functions

import tensorflow as tf
import tensorflow_probability as tfp

def cutmix_mask(alpha, h, w):
  """Returns image mask for CutMix."""
  r_x = tf.random.uniform([], 0, w, tf.int32)
  r_y = tf.random.uniform([], 0, h, tf.int32)

  area = tfp.distributions.Beta(alpha, alpha).sample()
  patch_ratio = tf.cast(tf.math.sqrt(1 - area), tf.float32)
  r_w = tf.cast(patch_ratio * tf.cast(w, tf.float32), tf.int32)
  r_h = tf.cast(patch_ratio * tf.cast(h, tf.float32), tf.int32)
  bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
  bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
  bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
  bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

  # Create the binary mask.
  pad_left = bbx1
  pad_top = bby1
  pad_right = tf.maximum(w - bbx2, 0)
  pad_bottom = tf.maximum(h - bby2, 0)
  r_h = bby2 - bby1
  r_w = bbx2 - bbx1

  mask = tf.pad(
      tf.ones((r_h, r_w)),
      paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
      mode='CONSTANT',
      constant_values=0)
  mask.set_shape((h, w))
  return mask[..., None]  # Add channel dim.

def cutmix(image, label, mask):
  """Applies CutMix regularization to a batch of images and labels.
  Reference: https://arxiv.org/pdf/1905.04899.pdf
  Arguments:
    image: a Tensor of batched images.
    label: a Tensor of batched labels.
    mask: a Tensor of batched masks.
  Returns:
    A new dict of features with updated images and labels with the same
    dimensions as the input with CutMix regularization applied.
  """
  # actual area of cut & mix pixels
  mix_area = tf.reduce_sum(mask) / tf.cast(tf.size(mask), mask.dtype)
  mask = tf.cast(mask, image.dtype)
  mixed_image = (1. - mask) * image + mask * image[::-1]
  mix_area = tf.cast(mix_area, label.dtype)
  mixed_label = (1. - mix_area) * label + mix_area * label[::-1]

  return mixed_image, mixed_label

def mixup(batch_size, alpha, image, label):
  """Applies Mixup regularization to a batch of images and labels.
  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412
  Arguments:
    batch_size: The input batch size for images and labels.
    alpha: Float that controls the strength of Mixup regularization.
    image: a Tensor of batched images.
    label: a Tensor of batch labels.
  Returns:
    A new dict of features with updated images and labels with the same
    dimensions as the input with Mixup regularization applied.
  """
  mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  img_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  image_type = image.dtype
  image = tf.cast(image, tf.float32)
  image = image * img_weight + image[::-1] * (1. - img_weight)
  image = tf.cast(image, image_type)

  label_type = label.dtype
  label = tf.cast(label, tf.float32)
  label = label * mix_weight + label[::-1] * (1 - mix_weight)
  label = tf.cast(label, label_type)

  return image, label

def mixing(batch_size, mixup_alpha, cutmix_alpha, features, labels):
  """Applies mixing regularization to a batch of images and labels.
  Arguments:
    batch_size: The input batch size for images and labels.
    mixup_alpha: Float that controls the strength of Mixup regularization.
    cutmix_alpha: FLoat that controls the strenght of Cutmix regularization.
    features: a dict of batched images.
    labels: a dict of batched labels.
  Returns:
    A new dict of features with updated images and labels with the same
    dimensions as the input.
  """
  image, label = features['image'], labels['label']
  if mixup_alpha and cutmix_alpha:
    # split the batch half-half, and aplly mixup and cutmix for each half.
    bs = batch_size // 2
    img1, lab1 = mixup(bs, mixup_alpha, image[:bs], label[:bs])
    img2, lab2 = cutmix(image[bs:], label[bs:],
                        features['cutmix_mask'][bs:])
    features['image'] = tf.concat([img1, img2], axis=0)
    labels['label'] = tf.concat([lab1, lab2], axis=0)
  elif mixup_alpha:
    features['image'], labels['label'] = mixup(batch_size, mixup_alpha,
                                               image, label)
  elif cutmix_alpha:
    features['image'], labels['label'] = cutmix(
        image, label, features['cutmix_mask'])
  return features['image'], labels['label']
