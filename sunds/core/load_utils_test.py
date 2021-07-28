# Copyright 2021 The Sunds Authors.
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

"""Tests for load_utils."""

from typing import Iterator

import pytest
import sunds
import tensorflow as tf
import tensorflow_datasets as tfds


class DummyTask(sunds.core.Task):
  """Dummy task which load the pipeline as-is."""

  def as_dataset(self, **kwargs):
    return self.frame_builder.as_dataset(**kwargs)


class DummyFrameTask(sunds.core.FrameTask):
  """Dummy frame task which only load the scene name."""

  @property
  def frame_specs(self):
    return {
        'scene_name': tfds.features.Tensor(shape=(), dtype=tf.string),
    }

  def pipeline(self, ds):
    ds = ds.batch(3)
    return ds


@pytest.fixture
def lego_data_dir(
    lego_builder: sunds.core.DatasetBuilder,
) -> Iterator[sunds.Path]:
  yield lego_builder._scene_builder.data_path.parent.parent.parent


def test_load(lego_data_dir: sunds.Path):  # pylint: disable=redefined-outer-name
  ds = sunds.load(
      'nerf_synthetic/lego',
      split='train',
      task=DummyTask(),
      data_dir=lego_data_dir,
  )
  assert isinstance(ds, tf.data.Dataset)
  assert 'scene_name' in ds.element_spec


def test_builder(lego_data_dir: sunds.Path):  # pylint: disable=redefined-outer-name
  builder = sunds.builder('nerf_synthetic/lego', data_dir=lego_data_dir)
  # `use_code == False`, so builder should be read-only
  assert isinstance(
      builder._scene_builder, tfds.core.read_only_builder.ReadOnlyBuilder)
  ds = builder.as_dataset(split='train', task=DummyTask())
  assert isinstance(ds, tf.data.Dataset)
  assert 'scene_name' in ds.element_spec


def test_frame_task(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(split='train', task=DummyFrameTask())
  assert isinstance(ds, tf.data.Dataset)
  # Only the scene name is loaded
  assert ds.element_spec == {
      'scene_name': tf.TensorSpec(shape=(None,), dtype=tf.string)
  }
