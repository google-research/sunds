# Copyright 2022 The sunds Authors.
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

"""Utils to encode the specs matching `sunds.specs`.

Used during data construction to create the example `dict` yield by
`_generate_examples`.

Using typed dataclass with explicitly named argument guarantee a structure
compatible with the corresponding `sunds.specs`.

There should be a one-to-one matching between `sunds.specs` and
`sunds.specs_utils`. (e.g.
`sunds.specs.frame_spec()` <> `sunds.specs_utils.Frame`).

"""

import dataclasses
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sunds.core import np_geometry
from sunds.core import specs_base

Array = Any  # numpy or any array-like
Image = Any  # str, np.array, file-object, pathlib object,...


@dataclasses.dataclass
class Scene(specs_base.SpecBase):
  """Top-level scene."""
  scene_name: str
  scene_box: Dict[str, Any]
  nominal_time: datetime.datetime = dataclasses.field(
      default_factory=lambda: datetime.datetime.utcfromtimestamp(0))
  frames: List['Frame'] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Frame(specs_base.SpecBase):
  """Top-level frame."""
  scene_name: str
  frame_name: str
  pose: 'Pose'
  cameras: Dict[str, 'Camera']
  timestamp: float = 0.0


# TODO(epot): Remove pose to use Isometry directly
@dataclasses.dataclass
class Pose(specs_base.SpecBase):
  """Pose."""
  R: Array  # pylint: disable=invalid-name
  t: Array

  @classmethod
  def from_transform_matrix(
      cls,
      transform_matrix: Array,
      *,
      convension: str,
  ) -> 'Pose':
    """Construct the pose frame wrt the scene (frame to scene transform)."""
    assert convension == 'blender'

    transform_matrix = np.array(transform_matrix)
    scene_from_bcam = np_geometry.Isometry.from_matrix(transform_matrix)
    # Transformation between blender style camera (+x along left to right,
    # +y from bottom to top, +z looking back) to camera system with +x along
    # left to right, +y from top to bottom, +z looking forward.
    bcam_from_frame = np_geometry.Isometry(
        R=np.array([[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0, 0, -1.0]]),
        t=np.zeros(3))
    scene_from_frame = scene_from_bcam.compose(bcam_from_frame)
    return cls(
        R=scene_from_frame.R.astype(np.float32),
        t=scene_from_frame.t.astype(np.float32),
    )

  @classmethod
  def identity(cls) -> 'Pose':
    """Returns the identity transformation."""
    return cls(
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
    )


@dataclasses.dataclass
class Camera(specs_base.SpecBase):
  """Camera."""
  intrinsics: 'CameraIntrinsics'
  extrinsics: Pose = dataclasses.field(default_factory=Pose.identity)
  color_image: Optional[Image] = None
  category_image: Optional[Image] = None
  ray_directions: Optional[Array] = None
  ray_origins: Optional[Array] = None


@dataclasses.dataclass
class CameraIntrinsics(specs_base.SpecBase):
  """CameraIntrinsics."""
  image_width: int
  image_height: int
  K: Array  # pylint: disable=invalid-name
  type: str
  distortion: Dict[str, Array]

  @classmethod
  def from_fov(
      cls,
      *,
      img_shape: Tuple[int, int],
      fov_in_degree: Tuple[int, int],
      type: str = 'PERSPECTIVE',  # pylint: disable=redefined-builtin
  ) -> 'CameraIntrinsics':
    """Returns camera instrinsics data."""
    if type != 'PERSPECTIVE':
      raise ValueError(
          f"Unknown camera type: {type!r}. Only 'PERSPECTIVE' supported")

    # Not sure if the height, width order is correct (shouldn't shape be (h, w)
    # instead ?
    camera_angle_x, camera_angle_y = fov_in_degree
    image_width, image_height = img_shape
    fx = .5 * image_width / np.tan(.5 * camera_angle_x)
    fy = .5 * image_height / np.tan(.5 * camera_angle_y)
    return cls(
        image_width=image_width,
        image_height=image_height,
        K=np.array(
            [
                [fx, 0, image_width / 2.0],  # fx 0 cx
                [0, fy, image_height / 2.0],  # 0, fy, cy
                [0, 0, 1],  # 0, 0, 1
            ],
            dtype=np.float32,
        ),
        type=type,
        distortion={
            'radial': np.zeros(3, dtype=np.float32),
            'tangential': np.zeros(2, dtype=np.float32),
        },
    )
