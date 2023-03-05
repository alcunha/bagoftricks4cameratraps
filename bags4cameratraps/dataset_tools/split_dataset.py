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

r"""Tool to split camera trap dataset into train/val/test sets.

This tool Splits a camera trap dataset using COCO json format according to
capture locations.
"""

import json

import pandas as pd
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_json', default=None,
    help=('Path to json file containing the full dataset info on COCO format.'))

flags.DEFINE_string(
    'dataset_split', default=None,
    help=('Path to json file containing the train/val/test split based on'
          ' locations.'))

flags.DEFINE_string(
    'location_key', default='location',
    help=('Key on json file containing the location id for each instance.'))

flags.mark_flag_as_required('dataset_json')
flags.mark_flag_as_required('dataset_split')

def main(_):
  with open(FLAGS.dataset_json) as json_file:
    metadata = json.load(json_file)
  images = pd.DataFrame(metadata['images'])
  annotations = pd.DataFrame(metadata['annotations'])

  with open(FLAGS.dataset_split) as json_file:
    splits = json.load(json_file)

  for split_name in splits.keys():
    split_json = FLAGS.dataset_json[:-len('.json')] + '_' + split_name + '.json'
    images_split = images[images[FLAGS.location_key].isin(splits[split_name])]
    ann_split = annotations[annotations['image_id'].isin(list(images_split.id))]

    metadata_split = metadata.copy()
    metadata_split['images'] = images_split.to_dict('records')
    metadata_split['annotations'] = ann_split.to_dict('records')

    print('Saving %s split to %s.' % (split_name, split_json))
    with open(split_json, 'w') as outfile_images:
      json.dump(metadata_split, outfile_images, indent=2)

if __name__ == '__main__':
  app.run(main)
