# Copyright 2023 The sunds Authors.
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

"""Functions to build the input pipeline."""

import abc
import copy
from typing import Any, Dict, Optional

from sunds.typing import FeatureSpecsHint, Split, Tree, TreeDict  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


class Task(abc.ABC):
  """Task abstract base class.

  A task is a simple object which control scene-understanding datasets pipeline.
  The `builder.as_dataset` call is directly forwarded to the task object, giving
  the task full control over the decoding.

  """

  scene_builder: tfds.core.DatasetBuilder
  frame_builder: tfds.core.DatasetBuilder

  def initialize(
      self,
      *,
      scene_builder: tfds.core.DatasetBuilder,
      frame_builder: tfds.core.DatasetBuilder,
  ) -> 'Task':
    """Lazy-initialization of the task."""
    # Tasks objects should be immutable, so we copy the original object
    # before initializing.
    task = copy.copy(self)
    # Use object.__setatrr__ to allow dataclass(frozen=True)
    object.__setattr__(task, 'scene_builder', scene_builder)
    object.__setattr__(task, 'frame_builder', frame_builder)
    return task

  @abc.abstractmethod
  def as_dataset(self, **kwargs) -> tf.data.Dataset:
    """Returns the `tf.data.Dataset` pipeline.

    Args:
      **kwargs: Kwargs to forwarded from `builder.as_dataset`.

    Returns:
       ds: The dataset
    """
    raise NotImplementedError('Abstract method')

  @property
  def full_scene_specs(self) -> tfds.features.FeaturesDict:
    """Nested structure of `tfds.features` of the frame dataset."""
    return self.frame_builder.info.features

  @property
  def full_frame_specs(self) -> tfds.features.FeaturesDict:
    """Nested structure of `tfds.features` of the scene dataset."""
    return self.frame_builder.info.features


class FrameTask(Task):
  """Frame task.

  Frame task allow to easily create pipelines based on the frame dataset.
  It requires to define:

   * `frame_specs`: Which subset of the feature specs to read.
   * `pipeline`: Which pre-processing to apply to the `tf.data.Dataset` object.

  ```python
  ds = sunds.load('nerf_synthetic', split='train', task=sunds.tasks.Nerf())
  for ex in ds:
    ex['rays']
  ```

  The final pipeline is more or less equivalent to:

  ```python
  builder = tfds.builder('nerf_synthetic_frames')
  ds = builder.as_dataset(
      decoders=tfds.features.PartialDecoding(task.frame_specs),
  )
  ds = task.pipeline(ds)
  ```

  """

  def as_dataset(
      self,
      *,
      split: Tree[Split],
      decoders: Optional[TreeDict[tfds.decode.Decoder]] = None,
      pipeline_kwargs: Optional[Dict[str, Any]] = None,
      **kwargs,
  ) -> tf.data.Dataset:
    """Returns the `tf.data.Dataset` pipeline.

    Args:
      split: The dataset split to load (e.g. 'train', 'train[80%:]',...)
      decoders: Optional decoders (e.g. to skip the image decoding)
      pipeline_kwargs: Additional kwargs to forward to the pipeline function.
      **kwargs: Kwargs to forward to `tfds.core.DatasetBuilder.as_dataset`.

    Returns:
       ds: The dataset
    """
    # Scene understanding datasets have many features. Only a very small subset
    # are required for a specific task.
    frame_specs = self.frame_specs
    if callable(frame_specs):
      raise ValueError(
          f'Invalid frame_specs={frame_specs!r}. Did you forgot `@property` ?'
      )
    if frame_specs is not None:
      decoders = tfds.decode.PartialDecoding(frame_specs, decoders=decoders)

    ds = self.frame_builder.as_dataset(
        split=split,
        decoders=decoders,
        **kwargs,
    )

    # Apply eventual post-processing on the dataset.
    return self.pipeline(ds, **(pipeline_kwargs or {}))

  @property
  @abc.abstractmethod
  def frame_specs(self) -> Optional[FeatureSpecsHint]:
    """Expected specs of the scene understanding pipeline.

    This should be a subset of the `full_frame_specs`.

    Returns:
      The nested structure of feature connector to decode. If `None`, the
        `full_frame_specs` will be decoded.
    """
    return None

  @abc.abstractmethod
  def pipeline(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Post processing step.

    This function allow the task to apply some additional post-processing to
    the.

    Args:
      ds: A dataset containing the `frame_specs` structure.

    Returns:
      ds: The dataset after the pipeline steps.
    """
    return ds
