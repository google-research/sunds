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

"""DatasetBuilder to create the frames."""

from typing import Any, Dict

from etils import epath
import numpy as np
from sunds.datasets.clevr3d import builder_utils
from sunds.datasets.clevr3d import utils
import tensorflow_datasets as tfds


class Clevr3dFrames(tfds.core.GeneratorBasedBuilder):
  """Frames dataset."""

  VERSION = builder_utils.VERSION
  RELEASE_NOTES = builder_utils.RELEASE_NOTES
  BUILDER_CONFIGS = builder_utils.CONFIGS
  TOTAL_NUM_VIEWS = builder_utils.NUM_CONDITIONAL_VIEWS + 1

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata."""

    frame_spec = builder_utils.get_frame_spec()

    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(frame_spec),
        **builder_utils.DATASET_INFO_KWARGS,
    )

  def _split_generators(
      self,
      dl_manager: tfds.download.DownloadManager,
  ) -> Dict[str, Any]:
    scene_dir = epath.Path(builder_utils.DATA_DIR)

    return {
        'train': self._generate_examples(
            split_name='train', scene_dir=scene_dir
        ),  # pytype: disable=wrong-arg-types  # gen-stub-imports
        'test': self._generate_examples(split_name='test', scene_dir=scene_dir),  # pytype: disable=wrong-arg-types  # gen-stub-imports
        'val': self._generate_examples(split_name='val', scene_dir=scene_dir),  # pytype: disable=wrong-arg-types  # gen-stub-imports
        'test7': self._generate_examples(
            split_name='test7', scene_dir=scene_dir
        ),  # pytype: disable=wrong-arg-types  # gen-stub-imports
    }

  def _generate_examples(self, scene_dir: epath.Path, split_name: str):
    # Load the metadata
    metadata_path = scene_dir / 'metadata.npz'
    metadata_dict = utils.load_metadata(metadata_path)

    # Get the dataset indices
    start_idx, end_idx = builder_utils.SPLIT_IDXs[split_name]

    num_objects = (metadata_dict['shape'][start_idx:end_idx] > 0).sum(1)

    max_num_objects = 10 if split_name == 'test7' else builder_utils.MAX_N
    min_num_objects = 7 if split_name == 'test7' else 3

    subset = (num_objects <= max_num_objects) & (num_objects >= min_num_objects)

    # Get the indices of the frames
    all_idxs = np.arange(start_idx, end_idx)[subset].tolist()

    beam = tfds.core.lazy_imports.apache_beam
    return beam.Create(all_idxs) | 'Generate Example' >> beam.Map(
        self._generate_single_example
    )

  def _generate_single_example(self, idx: int):
    scene_dir = epath.Path(builder_utils.DATA_DIR)
    rgb_dir = scene_dir / 'images'
    mask_dir = scene_dir / 'masks'

    # Get the conditioning views
    rgb_names = [
        rgb_dir / f'img_{idx}_{v}.png' for v in range(self.TOTAL_NUM_VIEWS)
    ]

    # Get the conditioning views
    mask_names = [
        mask_dir / f'masks_{idx}_{v}.png' for v in range(self.TOTAL_NUM_VIEWS)
    ]
    masks = utils.load_masks(mask_names)

    metadata_path = scene_dir / 'metadata.npz'
    metadata_dict = utils.load_metadata(metadata_path)
    scene_metadata = {k: v[idx] for (k, v) in metadata_dict.items()}

    # Get camera direction rays for the conditioning views
    rays_dir = []
    all_camera_pos = scene_metadata['camera_pos'][: self.TOTAL_NUM_VIEWS]
    for i in range(self.TOTAL_NUM_VIEWS):
      # TODO(klausg): Maybe use sunds auto-computation of rays from camera pos
      curr_rays = utils.get_camera_rays(all_camera_pos[i], noisy=False)
      rays_dir.append(curr_rays)
    rays_dir = np.stack(rays_dir, axis=0)
    all_camera_pos = np.expand_dims(
        np.expand_dims(all_camera_pos, axis=1), axis=1
    )
    all_camera_pos = np.tile(all_camera_pos, (1, 240, 320, 1))

    scene_name = f'{idx:04f}'
    frame_name = '0'  # single frame with 3 views/cameras

    # Prepare the frame data
    ray_origins = {
        'target_camera': all_camera_pos[0].astype(np.float32),
        'input_camera_0': all_camera_pos[1].astype(np.float32),
        'input_camera_1': all_camera_pos[2].astype(np.float32),
    }
    ray_dirs = {
        'target_camera': rays_dir[0].astype(np.float32),
        'input_camera_0': rays_dir[1].astype(np.float32),
        'input_camera_1': rays_dir[2].astype(np.float32),
    }
    color_image = {
        'target_camera': rgb_names[0],
        'input_camera_0': rgb_names[1],
        'input_camera_1': rgb_names[2],
    }
    mask_image = {
        'target_camera': masks[0],
        'input_camera_0': masks[1],
        'input_camera_1': masks[2],
    }
    frame = builder_utils.frame(
        scene_name=scene_name,
        frame_name=frame_name,
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        color_image=color_image,
        instance_image=mask_image,
    )
    return int(idx), frame.asdict()
