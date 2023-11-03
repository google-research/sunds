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

"""Utils to normalize scene boundaries."""

from typing import List, Tuple
from sunds.typing import ArrayDict, TensorLike  # pylint: disable=g-multiple-import
import tensorflow as tf


class MultiSceneBoundaries:
  """Lookup table across scene boundaries."""

  def __init__(self, scene_boundaries: List[ArrayDict]):
    """Constructor.

    Args:
      scene_boundaries: List of scene examples (as numpy arrays).
    """
    # Create the scene_name -> index id mapping
    scene_name2id = {
        ex['scene_name']: i for i, ex in enumerate(scene_boundaries)
    }
    self._table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            list(scene_name2id.keys()),
            list(scene_name2id.values()),
        ),
        # Ideally, it would be best if `StaticHashTable` supported not
        # having a default value, and raised a KeyError instead.
        default_value=-1,
    )

    # Stack all scenes values together
    corners = [ex['scene_box'] for ex in scene_boundaries]
    corners = tf.nest.map_structure(lambda *x: tf.stack(x), *corners)
    self._corners = corners

  def get_corners(self, scene_name: TensorLike) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns the scene boundaries for the requested scene.

    Args:
      scene_name: Scene name.

    Returns:
      min_corner, max_corner: The tuple (min, max) corners of the scene.
    """
    scene_name = tf.convert_to_tensor(scene_name, dtype=tf.string)
    scene_id = self._table.lookup(scene_name)

    # Make sure the scene was found.
    tf.debugging.assert_none_equal(scene_id, -1, message='Unknown scene name')

    scene_corners = tf.nest.map_structure(
        lambda x: tf.gather(x, scene_id),
        self._corners,
    )
    return scene_corners['min_corner'], scene_corners['max_corner']
