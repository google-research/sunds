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

"""Utils shared by frame_builder and scene_builder."""

import copy
import dataclasses
import gc
import textwrap
from typing import Dict, List, Optional, Tuple

from absl import logging
import numpy as np
import sunds
from sunds.datasets.kubric import base
from sunds.datasets.kubric import multi_shapenet
from sunds.typing import FeatureSpecs
import tensorflow_datasets as tfds

# Additional TFDS metadata
VERSION = tfds.core.Version('2.9.1')
RELEASE_NOTES = {
    '1.0.0': 'Initial version',
    '2.0.0': 'Migrated to MultiShapenet',
    '2.1.0': 'Also populate camera information.',
    '2.1.1': 'Fix camera information.',
    '2.2.0': 'Do not bake rays.',
    '2.2.1': '128x128 resolution for multishapenet conditional.',
    '2.3.0': 'Fix semantic maps.',
    '2.4.0': 'Added depth maps.',
    '2.5.0': 'Populate labels.',
    '2.6.0': 'Position cameras at least 2 units above ground.',
    '2.7.0': 'Move camera back.',
    '2.7.1': 'Fix splits.',
    '2.8.0': 'Export instance ids for objects (panoptic segmentation).',
    '2.9.0': 'Fix max num objects.',
    '2.9.1': 'Exclude objects with negative mass.',
}
DATASET_INFO_KWARGS = dict(
    homepage='https://srt-paper.github.io/',
    citation="""
    @article{srt22,
      title={{Scene Representation Transformer: Geometry-Free Novel View Synthesis Through Set-Latent Scene Representations}},
      author={Mehdi S. M. Sajjadi and Henning Meyer and Etienne Pot and Urs Bergmann and Klaus Greff and Noha Radwan and Suhani Vora and Mario Lucic and Daniel Duckworth and Alexey Dosovitskiy and Jakob Uszkoreit and Thomas Funkhouser and Andrea Tagliasacchi},
      journal={{CVPR}},
      year={2022},
      url={https://srt-paper.github.io/}
    }""",
)
_SEGMENTATION_LABEL_CACHE = {}


@dataclasses.dataclass
class KubricConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Kubric."""

  scene_config: base.SceneConfig = dataclasses.field(
      default_factory=multi_shapenet.SceneConfig
  )
  frames_per_scene: int = 10
  num_scenes: Dict[str, int] = dataclasses.field(
      default_factory=lambda: {'train': 6, 'test': 2}
  )
  is_conditional: bool = False

  def metadata(self) -> tfds.core.MetadataDict:
    return tfds.core.MetadataDict()

  def create_renderer(self) -> base.BaseRenderer:
    cfg = copy.copy(self.scene_config)
    cfg.frame_end = self.frames_per_scene
    if isinstance(self.scene_config, multi_shapenet.SceneConfig):
      return multi_shapenet.SceneRenderer(cfg)
    else:
      raise ValueError(f'Unknown ConfigType: {self.scene_config}')

  def camera_names(self) -> List[str]:
    if self.is_conditional:
      return [f'camera_{i}' for i in range(self.frames_per_scene)]
    else:
      return ['default']

# pylint: disable=unexpected-keyword-arg
BUILDER_CONFIGS = [
    KubricConfig(
        name='multi_shapenet',
        description=textwrap.dedent("""Basic MultiShapenet dataset."""),
        scene_config=multi_shapenet.SceneConfig(),
    ),
]
# pylint: enable=unexpected-keyword-arg


def frames_spec(
    img_shape: Tuple[int, int],
    camera_names: List[str],
    category_labels: Optional[List[str]] = None,
    has_image_features: bool = True,
) -> FeatureSpecs:
  """tf.Example specification for a single frame."""
  if has_image_features and category_labels:
    category_image = category_labels
  else:
    category_image = False
  camera_spec = sunds.specs.camera_spec(
      color_image=has_image_features,
      depth_image=has_image_features,
      category_image=category_image,
      instance_image=has_image_features,  # Object instance ids.
      img_shape=img_shape,
  )

  camera_specs = {camera_name: camera_spec for camera_name in camera_names}
  frame_spec = sunds.specs.frame_spec(cameras=camera_specs)
  # timestamp is not defined for this dataset.
  frame_spec.pop('timestamp')  # pytype: disable=attribute-error  # gen-stub-imports
  return frame_spec


def camera_intrinsics(
    img_shape: Tuple[int, int], focal_length: float, sensor_width: float
) -> sunds.specs_utils.CameraIntrinsics:
  camera_angle = np.arctan2(sensor_width / 2.0, focal_length) * 2
  return sunds.specs_utils.CameraIntrinsics.from_fov(
      img_shape=img_shape,
      fov_in_degree=(camera_angle * img_shape[0] / img_shape[1], camera_angle),
  )


def get_segmentation_labels(builder_config: KubricConfig) -> List[str]:
  """Retrieves the list of segmentation labels for a given config."""
  cfg_str = str(builder_config)
  if cfg_str in _SEGMENTATION_LABEL_CACHE:
    return _SEGMENTATION_LABEL_CACHE[cfg_str]
  gc.collect()
  logging.info('Creating renderer to fetch object categories.')
  renderer = builder_config.create_renderer()
  labels = renderer.segmentation_labels()
  _SEGMENTATION_LABEL_CACHE[cfg_str] = labels
  logging.info('Done fetching object categories.')
  return labels
