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

"""DatasetBuilder to create the scenes."""

import json
from typing import Any, Dict

import numpy as np
import sunds
from sunds.datasets.nerf_synthetic import builder_utils
import tensorflow_datasets as tfds


class NerfSyntheticScenes(tfds.core.GeneratorBasedBuilder):
  """Scenes dataset."""

  VERSION = builder_utils.VERSION
  RELEASE_NOTES = builder_utils.RELEASE_NOTES
  BUILDER_CONFIGS = builder_utils.CONFIGS

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata."""
    frames = sunds.specs.frame_spec(cameras={
        builder_utils.CAMERA_NAME: sunds.specs.camera_spec(),
    })
    features = sunds.specs.scene_spec(frames=frames)

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
            self._generate_examples(split_name='train', scene_dir=scene_dir),
        'test':
            self._generate_examples(split_name='test', scene_dir=scene_dir),
        'validation':
            self._generate_examples(split_name='val', scene_dir=scene_dir),
    }

  def _generate_examples(self, scene_dir: sunds.Path, split_name: str):
    scene_name = scene_dir.name
    scene_path = scene_dir / f'transforms_{split_name}.json'
    scene_data = json.loads(scene_path.read_text())

    intrinsics = builder_utils.camera_intrinsics(scene_data)

    def read_frame_info(frame_id, frame_data):
      return builder_utils.frame(
          scene_name=None,
          frame_name=builder_utils.frame_name(scene_name, split_name, frame_id),
          frame_data=frame_data,
          intrinsics=intrinsics,
      )

    frames = [
        read_frame_info(frame_id, frame_data)
        for frame_id, frame_data in enumerate(scene_data['frames'])
    ]
    yield scene_name, sunds.specs_utils.Scene(
        scene_name=scene_name,
        frames=frames,
        scene_box=_SCENE_BOXES[scene_name],
    ).asdict()  # TODO(epot): Could the asdict be automatically applied


# Scene bounding boxes extracted from the original blender files.
_SCENE_BOXES = {
    'chair': {
        'min_corner': np.array([-0.720808, -0.694973, -0.994077], dtype='f'),
        'max_corner': np.array([0.658137, 0.705611, 1.050102], dtype='f')
    },
    'drums': {
        'min_corner': np.array([-1.125537, -0.745907, -0.491643], dtype='f'),
        'max_corner': np.array([1.121641, 0.962200, 0.938314], dtype='f')
    },
    'ficus': {
        'min_corner': np.array([-0.377738, -0.857906, -1.033538], dtype='f'),
        'max_corner': np.array([0.555734, 0.577753, 1.140060], dtype='f')
    },
    'hotdog': {
        'min_corner': np.array([-1.197979, -1.286035, -0.189875], dtype='f'),
        'max_corner': np.array([1.197979, 1.109923, 0.311796], dtype='f')
    },
    'lego': {
        'min_corner': np.array([-0.637787, -1.140016, -0.344656], dtype='f'),
        'max_corner': np.array([0.633744, 1.148737, 1.002206], dtype='f')
    },
    'materials': {
        'min_corner': np.array([-1.122671, -0.758984, -0.231944], dtype='f'),
        'max_corner': np.array([1.071566, 0.985092, 0.199104], dtype='f'),
    },
    'mic': {
        'min_corner': np.array([-1.251289, -0.909447, -0.741352], dtype='f'),
        'max_corner': np.array([0.766763, 1.082312, 1.150916], dtype='f'),
    },
    'ship': {
        'min_corner': np.array([-1.276873, -1.299630, -0.549358], dtype='f'),
        'max_corner': np.array([1.370873, 1.348115, 0.728508], dtype='f'),
    },
}
