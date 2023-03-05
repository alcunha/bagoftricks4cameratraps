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

import json

import pandas as pd

def _load_images_data(train_json, category_map):
  with open(train_json) as json_file:
    json_data = json.load(json_file)
  images = pd.DataFrame(json_data['images'])
  annotations = pd.DataFrame(json_data['annotations'])
  images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')
  images['category_id'] = images['category_id'].apply(
                                          category_map.category_to_index)
  return images

def _get_bin(instances_count, sl_max_bin=[0, 10, 100, 1000, 2**100]):
  for group, group_max in enumerate(sl_max_bin):
    if instances_count < group_max:
      return group
  return 0

def get_cls_bin_lists(train_json, category_map):
  bins = {}
  images = _load_images_data(train_json, category_map)

  categories = list(range(category_map.get_num_classes()))
  for categ in categories:
    instances_count = len(images[images.category_id == categ])
    bin_id = _get_bin(instances_count)
    if bin_id in bins:
      bins[bin_id].append(categ)
    else:
      bins[bin_id] = [categ]

  return bins

def _get_metadata_from_model_path(model_path):
  exp_str = model_path.split('/')[-2]
  exp_data = exp_str.split('_')

  model = exp_data[0]
  dataset = exp_data[1]
  training_setup = exp_data[2]

  return model, dataset, training_setup

def save_results_to_json(data, json_file, model_path):
  model, dataset, training_setup = _get_metadata_from_model_path(model_path)
  data['model'] = model
  data['dataset'] = dataset
  data['training_setup'] = training_setup

  print(data)

  with open(json_file, 'w') as f:
    json.dump(data, f)
