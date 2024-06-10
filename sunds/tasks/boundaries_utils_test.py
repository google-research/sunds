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

"""Test of the scene boundaries."""

import numpy as np
import sunds
from sunds.tasks import boundaries_utils
import tensorflow_datasets as tfds


def test_scene_boundaries(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.scene_builder.as_dataset(split='train')
  scene_boundaries_values = list(tfds.as_numpy(ds))
  scene_boundaries = boundaries_utils.MultiSceneBoundaries(
      scene_boundaries_values
  )
  min_corner, max_corner = scene_boundaries.get_corners('lego')
  # Values from `datasets/nerf_synthetic/scene_builders.py`
  np.testing.assert_allclose(
      min_corner.numpy(),
      [-0.637787, -1.140016, -0.344656],
  )
  np.testing.assert_allclose(
      max_corner.numpy(),
      [0.633744, 1.148737, 1.002206],
  )
