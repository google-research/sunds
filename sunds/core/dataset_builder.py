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

"""Load API.

Primitive operations to load a dataset specs.
"""

import typing
from typing import Callable, Optional, Union

from etils import epath
from sunds.core import tasks as tasks_lib
from sunds.typing import Split, Tree  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds

_BuilderFactory = Callable[[], tfds.core.DatasetBuilder]


class LazyBuilder:
  """Lazy-DatasetBuilder.

  Behave like `tfds.core.DatasetBuilder`, but is only loaded on the first
  attribute call.
  """

  def __init__(
      self,
      builder_or_factory: Union[tfds.core.DatasetBuilder, _BuilderFactory],
  ):
    """Constructor."""
    if isinstance(builder_or_factory, tfds.core.DatasetBuilder):
      self._builder_instance = builder_or_factory
    else:
      self._builder_instance = None
      self._factory: _BuilderFactory = builder_or_factory
      if not callable(self._factory):
        raise ValueError('Either builder or factory should be given.')

  @property
  def _builder(self) -> tfds.core.DatasetBuilder:
    if self._builder_instance is None:
      # Lazy-loading
      self._builder_instance = self._factory()
    return self._builder_instance

  @property
  def loaded(self) -> bool:
    """Returns True if the builder has been loaded."""
    return self._builder_instance is not None

  def __getattr__(self, name: str):
    return getattr(self._builder, name)


class DatasetBuilder:
  """Wrapper around `tfds.core.DatasetBuilder`."""

  def __init__(
      self,
      *,
      scene_builder: LazyBuilder,
      frame_builder: LazyBuilder,
  ):
    """Constructor.

    Args:
      scene_builder: Dataset builder containing the scenes.
      frame_builder: Dataset builder containing the frames.
    """
    self._scene_builder = scene_builder
    self._frame_builder = frame_builder
    self._validate()

  def download_and_prepare(self, **kwargs):
    """Download and prepare the dataset."""
    # Is a no-op if the dataset has already been generated
    self.scene_builder.download_and_prepare(**kwargs)
    self.frame_builder.download_and_prepare(**kwargs)

  def as_dataset(
      self,
      *,
      split: Tree[Split],
      task: Optional[tasks_lib.Task] = None,
      **kwargs,
  ) -> tf.data.Dataset:
    """Returns the `tf.data.Dataset` pipeline.

    Args:
      split: The dataset split to load (e.g. 'train', 'train[80%:]',...)
      task: Task definition. Control the subset of spec read from the dataset,
        the pre-processing to apply,... If none, read the frames dataset.
      **kwargs: Kwargs to forward to `tfds.core.DatasetBuilder.as_dataset`.

    Returns:
      ds: The dataset
    """
    if task is None:
      import sunds  # pylint: disable=g-import-not-at-top

      task = sunds.tasks.Frames()
    task = task.initialize(
        scene_builder=self.scene_builder,
        frame_builder=self.frame_builder,
    )
    return task.as_dataset(split=split, **kwargs)

  @property
  def scene_builder(self) -> tfds.core.DatasetBuilder:
    """`tfds.core.DatasetBuilder` builder containing the scenes."""
    self._validate()
    return typing.cast(tfds.core.DatasetBuilder, self._scene_builder)

  @property
  def frame_builder(self) -> tfds.core.DatasetBuilder:
    """`tfds.core.DatasetBuilder` builder containing the frames."""
    self._validate()
    return typing.cast(tfds.core.DatasetBuilder, self._frame_builder)

  @property
  def data_dir(self) -> epath.Path:
    """Root directory of the frame/scene builder."""
    return epath.Path(self._frame_builder._data_dir_root)  # pylint: disable=protected-access

  def _validate(self) -> None:
    # Only validate if the 2 builders have been loaded.
    if not self._scene_builder.loaded or not self._frame_builder.loaded:
      return
    # Ensure that scene and frame datasets match
    if (
        self._scene_builder.info.full_name.split('/')[1:]
        != self._frame_builder.info.full_name.split('/')[1:]
    ):
      raise ValueError(
          f'Incompatible {self._scene_builder.info.full_name} and '
          f'{self._frame_builder.info.full_name}. Version and config should '
          'match.'
      )
