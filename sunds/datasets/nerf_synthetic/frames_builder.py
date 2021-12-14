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

"""DatasetBuilder to create the frames."""

import json
from typing import Any, Dict

import numpy as np
from PIL import Image
import sunds
from sunds.datasets.nerf_synthetic import builder_utils
import tensorflow_datasets as tfds


class NerfSyntheticFrames(tfds.core.GeneratorBasedBuilder):
  """Frames dataset."""

  VERSION = builder_utils.VERSION
  RELEASE_NOTES = builder_utils.RELEASE_NOTES
  BUILDER_CONFIGS = builder_utils.CONFIGS

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata."""
    features = sunds.specs.frame_spec(
        cameras={
            builder_utils.CAMERA_NAME:
                sunds.specs.camera_spec(
                    color_image=True,
                    img_shape=builder_utils.IMG_SHAPE,
                ),
        })

    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(features),
        **builder_utils.DATASET_INFO_KWARGS,
    )

  def _split_generators(
      self,
      dl_manager: tfds.download.DownloadManager,
  ) -> Dict[str, Any]:
    scene_dir = dl_manager.download_and_extract(builder_utils.DL_URL)
    scene_dir = scene_dir / 'nerf_synthetic' / self.builder_config.name

    return {
        'train':
            self._generate_examples(split_name='train', scene_dir=scene_dir),  # pytype: disable=wrong-arg-types  # gen-stub-imports
        'test':
            self._generate_examples(split_name='test', scene_dir=scene_dir),  # pytype: disable=wrong-arg-types  # gen-stub-imports
        'validation':
            self._generate_examples(split_name='val', scene_dir=scene_dir),  # pytype: disable=wrong-arg-types  # gen-stub-imports
    }

  def _generate_examples(self, scene_dir: sunds.Path, split_name: str):
    scene_name = scene_dir.name
    scene_path = scene_dir / f'transforms_{split_name}.json'
    scene_data = json.loads(scene_path.read_text())

    intrinsics = builder_utils.camera_intrinsics(scene_data)

    for frame_id, frame_data in enumerate(scene_data['frames']):
      img = _load_img_white_bkgd(scene_dir / split_name / f'r_{frame_id}.png')
      frame_name = builder_utils.frame_name(scene_name, split_name, frame_id)
      frame = builder_utils.frame(
          scene_name=scene_name,
          frame_name=frame_name,
          frame_data=frame_data,
          intrinsics=intrinsics,
          color_image=img,
      )
      # TODO(epot): Could the asdict be automatically applied ?
      yield frame_name, frame.asdict()


def _load_img_white_bkgd(path: sunds.Path) -> np.ndarray:
  """Load 4-channel images, returns 3-channel img with white background."""
  with path.open('rb') as f:
    img = np.array(Image.open(f))

  # Convert to RGB image with white Background.
  img = img.astype(np.float32) / 255.0
  img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
  img = (img * 255).astype(np.uint8)
  return img
