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

"""Helper utils for loading the data."""

from typing import Dict, List

from etils import epath
import imageio
import numpy as np


def get_camera_rays(
    c_pos,
    width=320,
    height=240,
    focal_length=0.035,
    sensor_width=0.032,
    noisy=False,
    vertical=None,
    c_track_point=None,
):
  """Compute the camera direction rays."""
  # The camera is pointed at the origin
  if c_track_point is None:
    c_track_point = np.array((0.0, 0.0, 0.0))

  if vertical is None:
    vertical = np.array((0.0, 0.0, 1.0))

  c_dir = c_track_point - c_pos
  c_dir = c_dir / np.linalg.norm(c_dir)

  img_plane_center = c_pos + c_dir * focal_length

  # The horizontal axis of the camera sensor is horizontal (z=0) and
  # orthogonal to the view axis
  img_plane_horizontal = np.cross(c_dir, vertical)
  img_plane_horizontal = img_plane_horizontal / np.linalg.norm(
      img_plane_horizontal
  )

  # The vertical axis is orthogonal to both the view axis and
  # the horizontal axis
  img_plane_vertical = np.cross(c_dir, img_plane_horizontal)
  img_plane_vertical = img_plane_vertical / np.linalg.norm(img_plane_vertical)

  # Double check that everything is orthogonal
  def is_small(x, atol=1e-7):
    return abs(x) < atol

  assert is_small(np.dot(img_plane_vertical, img_plane_horizontal))
  assert is_small(np.dot(img_plane_vertical, c_dir))
  assert is_small(np.dot(c_dir, img_plane_horizontal))

  # Sensor height is implied by sensor width and aspect ratio
  sensor_height = (sensor_width / width) * height

  # Compute pixel boundaries
  horizontal_offsets = np.linspace(-1, 1, width + 1) * sensor_width / 2
  vertical_offsets = np.linspace(-1, 1, height + 1) * sensor_height / 2

  # Compute pixel centers
  horizontal_offsets = (horizontal_offsets[:-1] + horizontal_offsets[1:]) / 2
  vertical_offsets = (vertical_offsets[:-1] + vertical_offsets[1:]) / 2

  horizontal_offsets = np.repeat(
      np.reshape(horizontal_offsets, (1, width)), height, 0
  )
  vertical_offsets = np.repeat(
      np.reshape(vertical_offsets, (height, 1)), width, 1
  )

  if noisy:
    pixel_width = sensor_width / width
    pixel_height = sensor_height / height
    horizontal_offsets += (
        np.random.random((height, width)) - 0.5
    ) * pixel_width
    vertical_offsets += (np.random.random((height, width)) - 0.5) * pixel_height

  horizontal_offsets = np.reshape(
      horizontal_offsets, (height, width, 1)
  ) * np.reshape(img_plane_horizontal, (1, 1, 3))
  vertical_offsets = np.reshape(
      vertical_offsets, (height, width, 1)
  ) * np.reshape(img_plane_vertical, (1, 1, 3))

  image_plane = horizontal_offsets + vertical_offsets

  image_plane = image_plane + np.reshape(img_plane_center, (1, 1, 3))
  c_pos_exp = np.reshape(c_pos, (1, 1, 3))
  rays = image_plane - c_pos_exp
  ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
  rays = rays / ray_norms
  return rays.astype(np.float32)


def depths_to_world_coords(
    depths, rays, camera_pos, depth_noise=None, noise_ratio=1.0
):
  """Project depth to world coordinates."""
  if depth_noise is not None:
    noise_indicator = (np.random.random(depths.shape) <= noise_ratio).astype(
        np.float32
    )
    depths = (
        depths + noise_indicator * np.random.random(depths.shape) * depth_noise
    )

  surface_points = camera_pos + rays * np.expand_dims(depths, -1)
  return surface_points.astype(np.float32)


def load_metadata(metadata_path: epath.Path) -> Dict[str, np.ndarray]:
  """Load the metadata as a dictionary."""
  with metadata_path.open('rb') as fh:
    metadata = np.load(fh)
    metadata = {k: v for k, v in metadata.items()}
    return metadata


def load_rgb_images(image_names: List[epath.Path]) -> List[np.ndarray]:
  """Loads the RGB images."""
  res = []
  for im_name in image_names:
    with im_name.open('rb') as rfh:
      rgb_img = np.asarray(imageio.imread(rfh))
      rgb_img = rgb_img[..., :3].astype(np.uint8)
      res.append(rgb_img)
  return res


def load_masks(image_names: List[epath.Path]) -> List[np.ndarray]:
  """Loads the segmentation images."""
  res = []
  for im_name in image_names:
    with im_name.open('rb') as rfh:
      mask_img = np.asarray(imageio.imread(rfh)) + 1
      res.append(mask_img[..., None])
  return res


def load_depth_images(depth_names: List[epath.Path]) -> List[np.ndarray]:
  """Loads the depth images."""
  res = []
  for im_name in depth_names:
    with im_name.open('rb') as dfh:
      depth_img = np.asarray(imageio.imread(dfh))
      # Convert 16 bit integer depths to floating point numbers.
      # 0.025 is the normalization factor used while drawing the depthmaps.
      depth_img = depth_img.astype(np.float32) / (65536 * 0.025)
      depth_img = np.expand_dims(depth_img, axis=-1)
      res.append(depth_img)
  return res
