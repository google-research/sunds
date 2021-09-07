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

"""Add a normal feature."""

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class NormalImage(tfds.features.Image):
  """Normal Image `FeatureConnector` for storing normal maps.

  Given a floating point normal image, the encoder internally stores the normal
  image as a 3 channel uint16 PNG image.

  Example:

  * In the DatasetInfo object:

    ```
    features=features.FeaturesDict({
        'normal': features.Normal(shape=(480, 640)),
    })
    ```

  * During generation:

    ```
    yield {
        'normal': np.random.rand(480, 640, 3).astype(np.float32),
    }
    ```

  * Decoding will return as dictionary of tensorflow tensors:

    ```
    {
        'normal': tf.Tensor(shape=(480, 640, 3), dtype=tf.float32)
    }
    ```

  """

  def __init__(
      self,
      shape: Tuple[tfds.typing.Dim, tfds.typing.Dim] = (None, None),
  ):
    """Constructor.

    Args:
      shape: (height, width) shape of the image
    """
    super().__init__(
        shape=(*shape, 3),
        encoding_format='png',
        dtype=tf.uint16,
    )
    self._scale = np.iinfo(np.uint16).max / 2.0

  def get_tensor_info(self) -> tfds.features.TensorInfo:
    return tfds.features.TensorInfo(shape=self._shape, dtype=tf.float32)

  def encode_example(self, normal_np: np.ndarray) -> bytes:
    normal_discrete = ((normal_np + 1.0) * self._scale).astype(np.uint16)
    return super().encode_example(normal_discrete)

  def decode_example(self, example: tf.Tensor) -> tf.Tensor:
    normal_discrete = super().decode_example(example)
    normal = (tf.cast(normal_discrete, tf.float32) / self._scale) - 1.0
    return normal

  @classmethod
  def from_json_content(cls, value: tfds.typing.Json) -> 'NormalImage':
    return cls(
        shape=value['shape'],  # pytype: disable=wrong-arg-types
    )

  def to_json_content(self) -> tfds.typing.Json:
    return dict(
        shape=self._shape[:2],
    )

  def repr_html(self, ex: np.ndarray) -> str:
    discretized = ((ex + 1.0) * self._scale).astype(np.uint8)
    return super().repr_html(discretized)
