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

"""Tests for isun.geometry.rays."""

from absl.testing import parameterized
import numpy as np
from sunds.core.tf_geometry import cameras
from sunds.core.tf_geometry import isometry
from sunds.core.tf_geometry import rays
import tensorflow as tf


def simple_pinhole_camera():
  image_width, image_height = 320, 240
  fx, fy, cx, cy = 400.0, 400.0, image_width / 2, image_height / 2
  return cameras.PinholeCamera(
      K=[[fx, 0., cx], [0., fy, cy], [0., 0., 1.]],
      image_width=image_width,
      image_height=image_height)


def random_pose():
  """Use QR decomposition of random (normal) matrix to get random rotation."""
  random_rotation, _ = tf.linalg.qr(tf.random.normal(shape=[3, 3]))
  return isometry.Isometry(R=random_rotation, t=tf.random.uniform(shape=[3]))


def pinhole_camera_with_random_pose():
  return {
      'testcase_name': 'pinhole_camera_with_random_pose',
      'camera': simple_pinhole_camera(),
      'world_from_camera': random_pose(),
  }


def pinhole_camera_with_non_square_pixels():
  camera = cameras.PinholeCamera.from_intrinsics(
      image_size_in_pixels=(1024, 768), focal_length_in_pixels=(500.0, 600.0))
  world_from_camera = isometry.Isometry(R=tf.eye(3), t=tf.ones(3))
  return {
      'testcase_name': 'pinhole_camera_with_non_square_pixels',
      'camera': camera,
      'world_from_camera': world_from_camera,
  }


class RaysFromImageGridTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(pinhole_camera_with_random_pose(),
                                  pinhole_camera_with_non_square_pixels())
  def test_rays_shape(self, camera, world_from_camera):
    """Test that the output tensor shapes are correct."""
    ray_origins, ray_directions = rays.rays_from_image_grid(
        camera=camera, world_from_camera=world_from_camera)
    expected_shape = (camera.image_height, camera.image_width, 3)
    self.assertEqual(ray_origins.shape, expected_shape)
    self.assertEqual(ray_directions.shape, expected_shape)

  @parameterized.named_parameters(pinhole_camera_with_random_pose(),
                                  pinhole_camera_with_non_square_pixels())
  def test_rays_are_normalized(self, camera, world_from_camera):
    """Test that each ray direction is an unit vector."""
    _, ray_directions = rays.rays_from_image_grid(
        camera=camera, world_from_camera=world_from_camera)
    self.assertAllClose(
        tf.norm(ray_directions, axis=-1),
        tf.ones(shape=(camera.image_height, camera.image_width)),
        atol=1e-04)

  @parameterized.named_parameters(pinhole_camera_with_random_pose(),
                                  pinhole_camera_with_non_square_pixels())
  def test_rays_project_back_to_image(self, camera, world_from_camera):
    """Make sure that points on the rays project to pixel centers on image."""
    # Generate rays.
    ray_origins, ray_directions = rays.rays_from_image_grid(
        camera=camera, world_from_camera=world_from_camera)

    # Generate ray end points in world frame and project back to camera.
    points_world = ray_origins + 5.0 * ray_directions
    points_camera = world_from_camera.inverse() * points_world
    points_image = camera.project(points_camera)

    # Create an array (H, W, 2) containing pixel center coordinates.
    pixel_centers = np.stack(
        np.meshgrid(
            range(camera.image_width),
            range(camera.image_height),
            indexing='xy'),
        axis=-1)
    pixel_centers = pixel_centers.astype(np.float32) + 0.5

    self.assertAllClose(points_image, pixel_centers, atol=1e-03)


class SamplesAlongRaysTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('method=linspace_depth', 'linspace_depth'),
      ('method=linspace_disparity', 'linspace_disparity'),
      ('method=geomspace_depth', 'geomspace_depth'),
  )
  def test_depth_samples_scalar_range(self, method):
    """Checks depth samples with a single scalar input depth range."""
    point_depths = rays.depth_samples_along_rays(
        near_depth=1.0, far_depth=10.0, num_samples_per_ray=20, method=method)

    # Check tensor shape.
    self.assertEqual(point_depths.shape, (20, 1))

    # Check dpeths within range.
    self.assertAllInRange(point_depths, lower_bound=1.0, upper_bound=10.0)

    # Check that depths are ordered as monotonically increasing order.
    self.assertTrue(
        tf.reduce_all(tf.experimental.numpy.diff(point_depths, axis=-1) > 0.0))

  @parameterized.named_parameters(
      ('method=linspace_depth', 'linspace_depth'),
      ('method=linspace_disparity', 'linspace_disparity'),
      ('method=geomspace_depth', 'geomspace_depth'),
  )
  def test_depth_samples_with_tensor_range(self, method):
    """Checks depth samples with multiple (tensor) depth ranges."""
    point_depths = rays.depth_samples_along_rays(
        near_depth=tf.ones(shape=(42, 420)),
        far_depth=10.0 * tf.ones(shape=(42, 420)),
        num_samples_per_ray=20,
        method=method)

    # Check tensor shape.
    self.assertEqual(point_depths.shape, (42, 420, 20, 1))

    # Check dpeths within range.
    self.assertAllInRange(point_depths, lower_bound=1.0, upper_bound=10.0)

    # Check that depths are ordered as monotonically increasing order.
    self.assertTrue(
        tf.reduce_all(tf.experimental.numpy.diff(point_depths, axis=-1) > 0.0))

  @parameterized.named_parameters(
      ('method=linspace_depth', 'linspace_depth'),
      ('method=linspace_disparity', 'linspace_disparity'),
      ('method=geomspace_depth', 'geomspace_depth'),
  )
  def test_jitter_depth_samples_histogram(self, method):
    """Tests that histogram of jittered depth samples are positive."""
    # With stratified sampling histogram of large number of depth samples will
    # cover all fixed width histogram bins in [near_depth, far_depth].

    # Set random seed for test repeatability.
    tf.random.set_seed(42)

    # Get depth samples.
    near_depth = 10.0
    far_depth = 100.0
    num_samples_per_ray = 10
    point_depths = rays.depth_samples_along_rays(
        near_depth=near_depth,
        far_depth=far_depth,
        num_samples_per_ray=num_samples_per_ray,
        method=method)
    point_depths = tf.broadcast_to(point_depths,
                                   [100000, num_samples_per_ray, 1])

    # Get jittered depth samples.
    jittered_point_depths = rays.jitter_depth_samples_along_rays(point_depths)

    # Get histogram of the depth samples and make sure all histogram bins > 0
    hist = tf.histogram_fixed_width(
        jittered_point_depths, value_range=[near_depth, far_depth], nbins=1000)
    self.assertAllGreater(hist, 0.0)

  @parameterized.named_parameters(
      ('method=linspace_depth', 'linspace_depth'),
      ('method=linspace_disparity', 'linspace_disparity'),
      ('method=geomspace_depth', 'geomspace_depth'),
  )
  def test_jittere_depth_samples_monotonicity(self, method):
    """Tests that jittered depth samples are monotonically increasing."""
    point_depths = tf.sort(
        tf.random.uniform(shape=[100, 1], minval=2.0, maxval=6.0), axis=-2)
    jittered_point_depths = rays.jitter_depth_samples_along_rays(point_depths)
    self.assertAllGreaterEqual(
        jittered_point_depths[1:, :] - jittered_point_depths[:-1, :], 0.0)

  @parameterized.parameters(
      {'point_depths_shape': [5, 1]},
      {'point_depths_shape': [4, 5, 1]},
      {'point_depths_shape': [3, 4, 5, 1]},
  )
  def test_jitter_depth_samples_shape(self, point_depths_shape):
    """Tests that jittered depth samples have same shape as input depths."""
    point_depths = tf.sort(
        tf.random.uniform(shape=point_depths_shape, minval=2.0, maxval=6.0),
        axis=-2)
    jittered_point_depths = rays.jitter_depth_samples_along_rays(point_depths)
    self.assertEqual(point_depths.shape, jittered_point_depths.shape)

  @parameterized.named_parameters(
      ('method=linspace_depth', 'linspace_depth'),
      ('method=linspace_disparity', 'linspace_disparity'),
      ('method=geomspace_depth', 'geomspace_depth'),
  )
  def test_point_samples_along_rays_within_range(self, method):
    """Generate point samples along rays and if their depths are in range."""
    near_depth = 10.0
    far_depth = 100.0
    num_samples_per_ray = 20
    point_depths = rays.depth_samples_along_rays(
        near_depth=near_depth,
        far_depth=far_depth,
        num_samples_per_ray=num_samples_per_ray,
        method=method)

    ray_origins = tf.zeros(3)
    ray_directions = tf.random.uniform(
        shape=(240, 320, 3), minval=-1.0, maxval=1.0)
    ray_directions, _ = tf.linalg.normalize(ray_directions, axis=-1)

    point_positions = rays.point_samples_along_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        point_depths=point_depths)
    point_depths = tf.linalg.norm(point_positions, axis=-1)

    self.assertEqual(point_positions.shape, (240, 320, num_samples_per_ray, 3))
    self.assertAllInRange(
        point_depths,
        lower_bound=near_depth - 1e-2,
        upper_bound=far_depth + 1e-2)
