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

"""DatasetBuilder to create the frames including conditional views."""

import binascii
import dataclasses
import gc
from typing import Any, Dict

from absl import logging
from jax3d.projects.nesf.nerfstatic.utils import camera_utils
import numpy as np
import sunds
from sunds.datasets.kubric import base
from sunds.datasets.kubric import utils
import tensorflow_datasets as tfds

_DEFAULT_NAMESPACE = 'KUBRIC'


class KubricFrames(tfds.core.GeneratorBasedBuilder):
  """Kubric Frames dataset.

  Dataset from synthetic object.
  """

  BUILDER_CONFIGS = utils.BUILDER_CONFIGS
  VERSION = utils.VERSION

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict(
            utils.frames_spec(
                img_shape=(
                    self.builder_config.scene_config.resolution[0],
                    self.builder_config.scene_config.resolution[1],
                ),
                camera_names=self.builder_config.camera_names(),
                category_labels=utils.get_segmentation_labels(
                    self.builder_config
                ),
            )
        ),
        metadata=self.builder_config.metadata(),
        **utils.DATASET_INFO_KWARGS,
    )

  def _split_generators(
      self,
      dl_manager: tfds.download.DownloadManager,
  ) -> Dict[str, Any]:
    split_names = self.builder_config.num_scenes.keys()
    return {k: self._generate_examples(split=k) for k in split_names}

  def _generate_examples(self, split: str):
    beam = tfds.core.lazy_imports.apache_beam
    return beam.Create(
        range(self.builder_config.num_scenes[split])
    ) | 'Render scene' >> beam.FlatMap(self._render_example, split)

  def _render_example(self, seed: int, split: str):
    beam = tfds.core.lazy_imports.apache_beam
    gc.collect()
    logging.info('Creating renderer.')
    beam.metrics.Metrics.counter(_DEFAULT_NAMESPACE, 'RenderAttempt').inc()
    try:
      renderer = self.builder_config.create_renderer()
      logging.info('Starting render.')
      scene_seed = binascii.crc32(f'{seed}_{split}'.encode('utf-8'))
      if split == 'val':
        # Kubric only supports train and test split for selecting objects and
        # background.
        # Use test split for val as well.
        split = 'test'
      result = renderer.render_scene(seed=scene_seed, split=split)
      beam.metrics.Metrics.counter(_DEFAULT_NAMESPACE, 'RenderSuccess').inc()
      logging.info('Render done.')
      del renderer
      render_data = result['render_data']
      metadata = result['metadata']
      num_frames = metadata['num_frames']
      camera_meta = result['camera']

      cameras = camera_utils.Camera.from_position_and_quaternion(
          positions=np.array(camera_meta['positions']),
          quaternions=np.array(camera_meta['quaternions']),
          resolution=metadata['resolution'],
          # Assume square pixels: width / sensor_width == height / sensor_height
          focal_px_length=(
              camera_meta['focal_length']
              * metadata['resolution'][1]
              / camera_meta['sensor_width']
          ),
          use_unreal_axes=False,
      )
      camera_matrix = cameras.px2world_transform

      assert camera_matrix.shape == (
          num_frames,
          4,
          4,
      ), f'{camera_matrix.shape} vs {num_frames}'

      rgb = render_data['rgba'][..., :3]
      # Assert there's not more than 256 classes or object instances per scene.
      max_semantic_class = np.max(render_data['segmentation'])
      max_object_id = np.max(render_data['instances'])
      if max_semantic_class > 255 or max_object_id > 255:
        raise ValueError(
            f'Maximum semantic class in scene: {max_semantic_class},'
            f' maximum object instance id: {max_object_id}. '
            'Maximum allowed: 255, due to uint8).'
        )
      semantic = render_data['segmentation'].astype(np.uint8)
      instances = render_data['instances'].astype(np.uint8)
      intrinsics = utils.camera_intrinsics(
          img_shape=metadata['resolution'],
          focal_length=camera_meta['focal_length'],
          sensor_width=camera_meta['sensor_width'],
      )

      cameras = []
      identity_pose = sunds.specs_utils.Pose.identity()
      scene_name = f'scene_{seed}'
      for frame_id in range(num_frames):
        c2w = sunds.core.np_geometry.Isometry.from_matrix(
            camera_matrix[frame_id])
        blender_converter = sunds.core.np_geometry.Isometry(
            R=np.array([[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0, 0, -1.0]]),
            t=np.zeros(3),
        )
        c2w = c2w.compose(blender_converter)
        w2c = c2w.inverse()
        pose = sunds.specs_utils.Pose(
            R=w2c.R.astype(np.float32), t=w2c.t.astype(np.float32)
        )
        cameras.append(
            {
                'extrinsics': dataclasses.asdict(pose),
                'intrinsics': dataclasses.asdict(intrinsics),
                'color_image': rgb[frame_id],
                'depth_image': render_data['depth'][frame_id],
                'category_image': semantic[frame_id],
                'instance_image': instances[frame_id],
            }
        )

      if self.builder_config.is_conditional:
        beam.metrics.Metrics.counter(_DEFAULT_NAMESPACE, 'Examples').inc()
        yield str(seed), {
            'scene_name': scene_name,
            'frame_name': scene_name,
            # We don't have a pose, but need to pass something.
            # TODO(tutmann): fix.
            'pose': dataclasses.asdict(identity_pose),
            'cameras': dict(zip(self.builder_config.camera_names(), cameras)),
        }
      else:
        for frame_id, camera in enumerate(cameras):
          frame_name = str(frame_id)
          beam.metrics.Metrics.counter(_DEFAULT_NAMESPACE, 'Examples').inc()
          yield frame_name, {
              'scene_name': scene_name,
              'frame_name': frame_name,
              # We don't have a pose, but need to pass something.
              # TODO(tutmann): fix.
              'pose': dataclasses.asdict(identity_pose),
              'cameras': {
                  self.builder_config.camera_names()[0]: camera,
              },
          }
    except base.IrrecoverableRenderingError as e:
      logging.info('Encountered irrecoverable rendering error.')
      logging.info(e)
      logging.info('Proceeding without this sample.')
