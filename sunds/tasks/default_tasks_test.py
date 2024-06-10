# Copyright 2024 The sunds Authors.
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

"""Tests for default_tasks."""

import sunds
import tensorflow as tf


def test_frames(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(split='train', task=sunds.tasks.Frames())
  assert isinstance(ds, tf.data.Dataset)
  assert 'scene_name' in ds.element_spec
  assert 'frame_name' in ds.element_spec


def test_scenes(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(split='train', task=sunds.tasks.Scenes())
  assert isinstance(ds, tf.data.Dataset)
  assert 'scene_name' in ds.element_spec
  assert 'frame_name' not in ds.element_spec
