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

"""Geometry functions related to 3D rays."""

from typing import Dict

from sunds.core.tf_geometry import cameras
from sunds.core.tf_geometry import isometry
import tensorflow as tf

TensorLike = tf.types.experimental.TensorLike

TensorDict = Dict[str, tf.Tensor]


def rays_from_image_points(
    camera: cameras.CameraType,
    world_from_camera: isometry.Isometry,
    points_image: TensorLike,
):
  """Create bundle of rays passing through camera image points.

  Given a camera model, camera pose, and a set of 2D image points on the camera,
  this function creates a bundle of 3D rays in world frame passing through those
  image points and camera center.

  See `rays_from_image_grid` for generating rays passing through centers of all
  pixels in the camera image.

  The image points are provided as a tensor of shape `(..., 2)`, stored in
  convention defined by the camera model. For example if the camera model is an
  instance of type `isun.geometry.PinholeCamera`, the image points should have
  the following convention: The top left corner of the image is `(0, 0)` and the
  bottom right corner is `(image_width, image_height)`. So the center of the top
  left corner pixel is `(0.5, 0.5)`.

  Args:
    camera: An instance of camera of type `isun.geometry.AbstractCamera`.
    world_from_camera: Pose of the camera w.r.t world represented by an isometry
      transform `[R | t]` that takes points to camera frame to world.
    points_image: A tensor of shape `(..., 2)` containing image points.

  Returns:
    ray_orgins: Tensor of shape `(..., 3)` containing origin of each ray.
    ray_directions: Tensor of shape `(..., 3)` containing (normalized) direction
      vectors of each ray.
  """
  # Get rays through each pixel center by camera unproject.
  points_camera = camera.unproject(points_image)

  # Transform into world-frame rays.
  return _rays_from_directions(points_camera, world_from_camera)


def rays_from_image_grid(
    camera: cameras.CameraType, world_from_camera: isometry.Isometry
):
  """Create bundle of rays passing though all pixels of a camera image.

  Given a camera model, camera pose, this function creates a bundle of 3D rays
  in world frame passing through those each pixel in the camera image.

  Args:
    camera: An instance of camera of type `isun.geometry.AbstractCamera`.
    world_from_camera: Pose of the camera w.r.t world represented by an isometry
      transform `[R | t]` that takes points to camera frame to world.

  Returns:
    ray_orgins: Tensor of shape `(image_height, image_width, 3)` containing
      origin of each ray.
    ray_directions: Tensor of shape `(image_height, image_width, 3)` containing
      (normalized) direction vectors of each ray.
  """
  return _rays_from_directions(camera.unproject(), world_from_camera)


def _rays_from_directions(
    ray_directions: TensorLike, world_from_camera: isometry.Isometry
):
  """Convert rays in camera frame to world frame, now including their origins.

  Args:
    ray_directions: Tensor of shape `(..., 3)` containing (normalized) direction
      vectors of each ray in the camera frame.
    world_from_camera: Pose of the camera w.r.t world represented by an isometry
      transform `[R | t]` that takes points to camera frame to world.

  Returns:
    ray_orgins: Tensor of shape `(..., 3)` containing origin of each ray.
    ray_directions: Tensor of shape `(..., 3)` containing (normalized) direction
    vectors of each ray.
  """
  points_world = world_from_camera * ray_directions
  ray_origins = tf.broadcast_to(world_from_camera.t, tf.shape(points_world))
  ray_directions, _ = tf.linalg.normalize(points_world - ray_origins, axis=-1)

  return ray_origins, ray_directions


def depth_samples_along_rays(
    near_depth: TensorLike,
    far_depth: TensorLike,
    num_samples_per_ray: int,
    method: str = 'linspace_depth',
) -> tf.Tensor:
  """Sample depths along rays.

  This function samples depth values in the range [near_depth, far_depth]. The
  depth samples will include the end points near_depth, and far_depth; and will
  be in an increasing order.

  Supported methods are 'linspace_depth' (default), 'linspace_disparity' and
  'geomspace_depth'. If set to 'linspace_depth', samples are linearly spaced in
  the depth space. If set to 'linspace_disparity', the depth samples are spaced
  linearly in disparity space. With 'geomspace_depth', depth samples are evenly
  spaced in log space (a geometric progression).

  The depth extremas near_depth and far_depth can be arbritrary shaped tensor
  including scalar. However they should have the same shape. The output depth
  samples tensor will have a (..., num_samples_per_ray, 1), where the leading
  dimensions are same as that of near_depth and far_depth tensors.

  Args:
    near_depth: Near depth extrema. This can be arbritrary a shaped tensor
      including scalar.
    far_depth: Near depth extrema. This can be arbritrary shaped tensor
      including scalar.
    num_samples_per_ray: Number of samples per ray.
    method: Method to use for sampling. Supported methods are 'linspace_depth'
      (default), 'linspace_disparity' and 'geomspace_depth'.

  Returns:
    Tensor with shape (..., num_samples_per_ray, 1) containing depth samples. If
    the rank of the input tensors near_depth and far_depth is `r`, then rank of
    the `r + 2`. So when near_depth and far_depth are scalars, then output
    is a tensor of shape (num_samples_per_ray, 1).
  """
  near_depth = tf.convert_to_tensor(near_depth)
  far_depth = tf.convert_to_tensor(far_depth)

  tf.debugging.assert_equal(tf.shape(near_depth), tf.shape(far_depth))
  tf.debugging.assert_positive(near_depth)
  tf.debugging.assert_positive(far_depth)
  tf.debugging.assert_positive(far_depth - near_depth)

  if method == 'linspace_depth':
    depths = tf.linspace(near_depth, far_depth, num_samples_per_ray, axis=-1)
  elif method == 'linspace_disparity':
    depths = 1.0 / tf.linspace(
        1.0 / far_depth, 1.0 / near_depth, num_samples_per_ray, axis=-1
    )
    depths = tf.reverse(depths, axis=[-1])
  elif method == 'geomspace_depth':
    depths = tf.experimental.numpy.geomspace(
        near_depth,
        far_depth,
        num=num_samples_per_ray,
        endpoint=True,
        dtype=near_depth.dtype,
        axis=-1,
    )
  else:
    raise NotImplementedError(f'Unknown method: {method}')
  return tf.expand_dims(depths, axis=-1)


def jitter_depth_samples_along_rays(point_depths: TensorLike) -> tf.Tensor:
  """Jitters depth samples along rays using stratified sampling.

  Given a tensor containing depth samples along rays, this functions jitters the
  depth samples using stratified sampling. Jittered samples are drawn uniformly
  from the bins defined by input depth samples.

  Input depth samples should be ordered front-to-back: `point_depths[..., 0, :]`
  is the nearest sample on the ray, while `point_depths[..., -1, :]` is the
  farthest. In other words, `point_depths` values should be monotonically
  increasing along the penultimate axis.

  Example usage:

  ```python
  # Sample depth values.
  point_depths = isun.geometry.depth_samples_along_rays(
    near_depth=1.,
    far_depth=8.,
    num_samples_per_ray=100,
    method='geomspace_depth')

  # Broadcast `point_depths` for all rays in an image of shape (H, W).
  point_depths = tf.broadcast_to(point_depths, [H, W, 100, 1])

  # Jitter `point_depths` for each ray independently.
  point_depths = isun.geometry.jitter_depth_samples_along_rays(point_depths)
  ```

  Args:
    point_depths: Tensor of shape (..., num_samples_per_ray, 1) containing depth
      samples per ray.

  Returns:
    Tensor with shape (..., num_samples_per_ray, 1) containing jittered depths.
  """
  # Input checking.
  point_depths = tf.convert_to_tensor(point_depths)
  tf.debugging.assert_shapes([(point_depths, (..., 'num_samples_per_ray', 1))])

  # Get min, center, and max of each depth bin.
  bin_centers = (point_depths[..., 1:, :] + point_depths[..., :-1, :]) / 2
  bin_minimas = tf.concat([point_depths[..., :1, :], bin_centers], axis=-2)
  bin_maximas = tf.concat([bin_centers, point_depths[..., -1:, :]], axis=-2)

  # Get jittered depth samples.
  unscaled_jitter = tf.random.uniform(shape=tf.shape(point_depths))
  return bin_minimas + (bin_maximas - bin_minimas) * unscaled_jitter


def point_samples_along_rays(
    ray_origins: TensorLike,
    ray_directions: TensorLike,
    point_depths: TensorLike,
) -> tf.Tensor:
  """Sample points along rays.

  This function samples 3D points along rays with provided depth values.

  The rays are defined by ray_origins and ray_directions each of which should be
  tensors of shape (..., 3). It is possible that some rays share the same origin
  e.g. a ray_directions is shaped (H, W, 3) wheres as ray_origins is shaped(3,).
  However ray_origins should be broadcastable to the shape of ray_directions.

  The depth values at which the 3D points need to sampled are provided bt the
  point_depths tensor of shape (..., num_samples_per_ray, 1). Similar to
  ray_orgins it is possible that many rays share the same depth samples or to
  have a different depth samples for every ray. We expect this tensor to be
  generated by methods like `depth_samples_along_rays` possibly along with some
  user defined jittering.

  Example Usage:

  # Sample depth values.
  point_depths = depth_samples_along_rays(
      near_depth=1., far_depth=8., num_samples_per_ray=100, method='uniform')

  # Add jitter using some distribution.
  point_depths = point_depths + tf.random.normal(tf.shape(point_depths),
                                                 stddev=0.01)

  # Get 3D points along rays.
  point_positions = point_samples_along_rays(
      ray_origins=tf.zeros(3),
      ray_directions=tf.ones(shape=(240, 320, 3)),
      point_depths=point_depths)

  Args:
    ray_origins: Tensor of shape (..., 3) containing origin of each ray. This
      tensor should be broadcastable to the shape of ray_directions.
    ray_directions: Tensor of shape (..., 3) containing normalized direction
      vector of each ray.
    point_depths: Tensor of shape (..., num_samples_per_ray, 1) containing depth
      samples per ray.

  Returns:
    Tensor with shape (..., num_samples_per_ray, 3) containing 3D point samples
    along each ray.
  """
  tf.debugging.assert_shapes(
      [
          (ray_origins, (..., 3)),
          (ray_directions, (..., 3)),
          (point_depths, (..., 'num_samples_per_ray', 1)),
      ]
  )
  ray_origins = tf.expand_dims(ray_origins, axis=-2)
  ray_directions = tf.expand_dims(ray_directions, axis=-2)
  return ray_origins + point_depths * ray_directions
