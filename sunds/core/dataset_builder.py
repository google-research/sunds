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

"""Load API.

Primitive operations to load a dataset specs.
"""

from sunds import utils
from sunds.core import tasks as tasks_lib
from sunds.typing import Split, Tree  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetBuilder:
  """Wrapper around `tfds.core.DatasetBuilder`."""

  def __init__(
      self,
      *,
      scene_builder: tfds.core.DatasetBuilder,
      frame_builder: tfds.core.DatasetBuilder,
  ):
    """Constructor.

    Args:
      scene_builder: Dataset builder containing the scenes.
      frame_builder: Dataset builder containing the frames.
    """
    # Ensure that scene and frame datasets match
    if (
        scene_builder.info.full_name.split('/')[1:]
        != frame_builder.info.full_name.split('/')[1:]
    ):
      raise ValueError(
          f'Incompatible {scene_builder.info.full_name} and '
          f'{frame_builder.info.full_name}. Version and config should match.'
      )

    self._scene_builder = scene_builder
    self._frame_builder = frame_builder

  def download_and_prepare(self, **kwargs):
    """Download and prepare the dataset."""
    # Is a no-op if the dataset has already been generated
    self._scene_builder.download_and_prepare(**kwargs)
    self._frame_builder.download_and_prepare(**kwargs)

  def as_dataset(
      self,
      *,
      split: Tree[Split],
      task: tasks_lib.Task,
      **kwargs,
  ) -> tf.data.Dataset:
    """Returns the `tf.data.Dataset` pipeline.

    Args:
      split: The dataset split to load (e.g. 'train', 'train[80%:]',...)
      task: Task definition. Control the subset of spec read from the dataset,
        the pre-processing to apply,...
      **kwargs: Kwargs to forward to `tfds.core.DatasetBuilder.as_dataset`.

    Returns:
      ds: The dataset
    """
    task = task.initialize(
        scene_builder=self._scene_builder,
        frame_builder=self._frame_builder,
    )
    return task.as_dataset(split=split, **kwargs)

  @property
  def scene_builder(self) -> tfds.core.DatasetBuilder:
    """`tfds.core.DatasetBuilder` builder containing the scenes."""
    return self._scene_builder

  @property
  def frame_builder(self) -> tfds.core.DatasetBuilder:
    """`tfds.core.DatasetBuilder` builder containing the frames."""
    return self._frame_builder

  @property
  def data_dir(self) -> utils.Path:
    """Root directory of the frame/scene builder."""
    return utils.Path(self._frame_builder._data_dir_root)  # pylint: disable=protected-access
