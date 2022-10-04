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

"""Utils shared by frame_builder and scene_builder."""

from typing import Any, Dict, Optional

import numpy as np
import sunds
import tensorflow_datasets as tfds


# Additional TFDS metadata
RELEASE_NOTES = {
    '1.4.0': 'added test-set with 7-10 objects',
    '1.3.0': 'add one to segmentation masks to stay consistent with MSN_hard',
    '1.2.0': 'added segmentation masks instance_image',
    '1.1.0': 'remove depth map add placeholder poses',
    '1.0.0': 'Initial version',
}
VERSION = tfds.core.Version('1.4.0')


CONFIGS = [tfds.core.BuilderConfig(name='clevr3d')]

DATASET_INFO_KWARGS = dict(
    description="""CLEVR-3D Dataset.""",
    homepage='https://stelzner.github.io/obsurf',
    citation="""
    @article{stelzner2021decomposing,
      title={Decomposing 3d scenes into objects via unsupervised volume segmentation},
      author={Stelzner, Karl and Kersting, Kristian and Kosiorek, Adam R},
      journal={arXiv:2104.01148},
      year={2021}
    }
    """,
)
DATA_DIR = 'gs://kubric-unlisted/data/clevr3d/'
IMG_SHAPE = (240, 320)

CAMERA_NAMES = [
    'target_camera',
    'input_camera_0',
    'input_camera_1',
]

NUM_CONDITIONAL_VIEWS = 2

# maximum number of objects to use (filter the rest out)
MAX_N = 6

SPLIT_IDXs = {
    'train': (0, 70000),
    'val': (70000, 70500),
    'test': (85000, 100000),
    'test7': (0, 100000),
}


CAM = sunds.specs_utils.CameraIntrinsics(
    image_width=320,
    image_height=240,
    K=np.array(
        [
            [1.0, 0.0, 160.0],  # TODO(klausg): use actual intrinsics
            [0.0, 1.0, 120.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    type='PERSPECTIVE',
    distortion={
        'radial': np.zeros(3, dtype=np.float32),
        'tangential': np.zeros(2, dtype=np.float32),
    },
)


def frame(
    *,
    scene_name: str,
    frame_name: str,  # pylint: disable=redefined-outer-name
    ray_origins: Dict[str, np.ndarray],
    ray_directions: Dict[str, np.ndarray],
    color_image: Optional[Dict[str, Any]] = None,
    depth_image: Optional[Dict[str, Any]] = None,
    instance_image: Optional[Dict[str, Any]] = None,
) -> sunds.specs_utils.Frame:
  """Extract the frame information."""

  cameras = {}
  for camera_name in CAMERA_NAMES:
    cameras[camera_name] = sunds.specs_utils.Camera(
        intrinsics=CAM,
        color_image=color_image[camera_name] if color_image else None,
        depth_image=depth_image[camera_name] if depth_image else None,
        instance_image=instance_image[camera_name] if instance_image else None,
        # TODO(klausg): use actual extrinsics
        ray_origins=ray_origins[camera_name],
        ray_directions=ray_directions[camera_name],
    )
  return sunds.specs_utils.Frame(
      scene_name=scene_name,
      frame_name=frame_name,
      cameras=cameras,
      pose=sunds.specs_utils.Pose.identity(),  # TODO(klausg): use actual pose
  )


def get_frame_spec():
  """Defines the frame spec."""

  camera_spec = sunds.specs.camera_spec(
      color_image=True,
      depth_image=False,
      category_image=False,
      instance_image=True,  # Object instance ids.
      camera_rays=True,
      img_shape=IMG_SHAPE,
  )

  camera_specs = {camera_name: camera_spec for camera_name in CAMERA_NAMES}
  frame_spec = sunds.specs.frame_spec(cameras=camera_specs)
  return frame_spec
