# Copyright 2021 The sunds Authors.
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

"""Default frame and scene tasks."""

from sunds import core
import tensorflow as tf


class Frames(core.Task):
  """Simple task which loads the frames."""

  def as_dataset(self, **kwargs) -> tf.data.Dataset:
    """Returns the `tf.data.Dataset` pipeline.

    Args:
      **kwargs: Kwargs to forwarded from `builder.as_dataset`.

    Returns:
       ds: The dataset
    """
    return self.frame_builder.as_dataset(**kwargs)


class Scenes(core.Task):
  """Simple task which loads the scenes."""

  def as_dataset(self, **kwargs) -> tf.data.Dataset:
    """Returns the `tf.data.Dataset` pipeline.

    Args:
      **kwargs: Kwargs to forwarded from `builder.as_dataset`.

    Returns:
       ds: The dataset
    """
    return self.scene_builder.as_dataset(**kwargs)
