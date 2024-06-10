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

"""Tests for normal."""

import numpy as np
from sunds.features import normal
import tensorflow as tf
import tensorflow_datasets as tfds


def _get_encoded_img(image_array: np.ndarray) -> bytes:
  if image_array.ndim == 2:  # Expand 2d array
    image_array = image_array[..., None]
  return tf.image.encode_png(image_array).numpy()


class ImageFeaturesTest(tfds.testing.FeatureExpectationsTestCase):

  def testNormalImage(self):
    img = np.random.rand(24, 24, 3).astype(np.float32)
    img_discrete = ((img + 1.0) * 32767.5).astype(np.uint16)
    img2 = np.array(
        [
            [[1.0, 0.0, 0.0]],
            [[1e-6, 0.4, -1.0]],
            [[0.9999999999999999999999999999, 0.4, -1.0]],
        ]
    )
    atol = 1e-04

    self.assertFeatureEagerOnly(
        feature=normal.NormalImage(),
        shape=(None, None, 3),
        dtype=np.float32,
        tests=[
            # Numpy array
            tfds.testing.FeatureExpectationItem(
                value=img,
                expected=img,
                expected_serialized=_get_encoded_img(img_discrete),
                atol=atol,
            ),
            tfds.testing.FeatureExpectationItem(
                value=img2,
                expected=img2,
                atol=atol,
            ),
        ],
    )
