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

"""Utils shared by frame_builder and scene_builder."""

from typing import Optional

import numpy as np
import sunds
from sunds.typing import Json
import tensorflow_datasets as tfds

SCENE_NAMES = [
    'chair',
    'drums',
    'ficus',
    'hotdog',
    'lego',
    'materials',
    'mic',
    'ship',
]

# Additional TFDS metadata
VERSION = tfds.core.Version('1.0.0')
RELEASE_NOTES = {
    '1.0.0': 'Initial version',
}
CONFIGS = [
    tfds.core.BuilderConfig(name=scene_name) for scene_name in SCENE_NAMES
]
DATASET_INFO_KWARGS = dict(
    description="""Nerf synthetic dataset.""",
    homepage='https://github.com/bmild/nerf',
    citation="""
    @inproceedings{mildenhall2020nerf,
      title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
      author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
      booktitle={European Conference on Computer Vision},
      pages={405--421},
      year={2020},
      organization={Springer}
    }
    """,
)
DL_URL = 'https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG&export=download'

IMG_SHAPE = (800, 800)

CAMERA_NAME = 'default_camera'


def camera_intrinsics(scene_data: Json) -> sunds.specs_utils.CameraIntrinsics:
  return sunds.specs_utils.CameraIntrinsics.from_fov(
      img_shape=IMG_SHAPE,
      fov_in_degree=(scene_data['camera_angle_x'],
                     scene_data['camera_angle_x']),
  )


def frame(
    *,
    scene_name: Optional[str] = None,
    frame_name: str,  # pylint: disable=redefined-outer-name
    frame_data: Json,
    intrinsics: sunds.specs_utils.CameraIntrinsics,
    color_image: Optional[np.ndarray] = None,
) -> sunds.specs_utils.Frame:
  """Extract the frame information."""
  pose = sunds.specs_utils.Pose.from_transform_matrix(
      frame_data['transform_matrix'],
      convension='blender',
  )
  camera = sunds.specs_utils.Camera(
      intrinsics=intrinsics, color_image=color_image)
  return sunds.specs_utils.Frame(
      scene_name=scene_name,
      frame_name=frame_name,
      pose=pose,
      cameras={CAMERA_NAME: camera},
  )


def frame_name(scene_name: str, split_name: str, frame_id: int) -> str:
  return f'{scene_name}_{split_name}_frame{frame_id:04}'
