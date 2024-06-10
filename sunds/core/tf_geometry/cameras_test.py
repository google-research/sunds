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

"""Tests for isun.geometry.cameras."""

import numpy as np
from sunds.core.tf_geometry import cameras
import tensorflow as tf


class PinholeCameraTest(tf.test.TestCase):

  def test_construct_from_fov(self):
    """Construct camera using field of view and verify camera attributes."""
    camera_model = cameras.PinholeCamera.from_fov(
        image_width=640,
        image_height=480,
        horizontal_fov_in_degrees=57.5,
        vertical_fov_in_degrees=45.0,
    )
    self.assertEqual(camera_model.image_width, 640)
    self.assertEqual(camera_model.image_height, 480)
    self.assertAllClose(
        camera_model.K,
        [[583.28296, 0.0, 320.0], [0.0, 579.41125, 240.0], [0.0, 0.0, 1.0]],
    )

  def test_construct_from_intrinsics(self):
    """Construct camera using intrinsics params and verify camera attributes."""
    camera_model = cameras.PinholeCamera.from_intrinsics(
        image_size_in_pixels=(640, 480),
        focal_length_in_pixels=(500.0, 400.0),
        principal_point_in_pixels=(320.1, 239.9),
        skew=0.1,
    )
    self.assertEqual(camera_model.image_width, 640)
    self.assertEqual(camera_model.image_height, 480)
    self.assertAllClose(
        camera_model.K,
        [[500.0, 0.1, 320.1], [0.0, 400.0, 239.9], [0.0, 0.0, 1.0]],
    )

  def test_project_principal_point_single(self):
    """3D points on +Z axis should project to principal point of the camera."""
    image_width, image_height = 640, 480
    fx, fy, cx, cy = 500.0, 500.0, image_width / 2, image_height / 2
    camera_model = cameras.PinholeCamera(
        K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        image_width=image_width,
        image_height=image_height,
    )

    points_camera = np.array([0.0, 0.0, 1.0])
    points_image = camera_model.project(points_camera)
    self.assertEqual(points_image.shape, (2,))
    self.assertAllClose(points_image, [cx, cy])

  def test_project_principal_point_batched(self):
    """3D points on +Z axis should project to principal point of the camera."""
    image_width, image_height = 640, 480
    fx, fy, cx, cy = 500.0, 500.0, image_width / 2, image_height / 2
    camera_model = cameras.PinholeCamera(
        K=tf.constant([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        image_width=image_width,
        image_height=image_height,
    )

    points_camera = np.tile(np.array([0.0, 0.0, 1.0]), (10, 5, 1))
    points_image = camera_model.project(points_camera)
    self.assertEqual(points_image.shape, (10, 5, 2))
    self.assertAllClose(points_image, np.tile([cx, cy], (10, 5, 1)))

  def test_project_points(self):
    """Tests projection of random set of 3D points."""
    num_points = 10
    points_camera = np.random.rand(num_points, 3)
    intrinsics = np.array(
        [[500, 0.0, 320.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]]
    )
    camera_model = cameras.PinholeCamera(
        K=intrinsics, image_width=640, image_height=240
    )
    points_image = camera_model.project(points_camera)
    self.assertEqual(points_image.shape, (num_points, 2))
    for point_camera, point_image in zip(points_camera, points_image):
      expected = intrinsics.dot(point_camera)
      expected = expected[:2] / expected[2]
      self.assertAllClose(point_image, expected)

  def test_unproject_principal_point_single(self):
    """Principal point should unproject to camera space points along +Z."""
    image_width, image_height = 640, 480
    fx, fy, cx, cy = 500.0, 500.0, image_width / 2, image_height / 2
    camera_model = cameras.PinholeCamera(
        K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        image_width=image_width,
        image_height=image_height,
    )

    points_image = np.array([cx, cy])
    points_camera = camera_model.unproject(points_image)
    self.assertAllClose(points_camera, [0.0, 0.0, 1.0])

  def test_unproject_principal_point_batched(self):
    """Principal point should unproject to camera space points along +Z."""
    image_width, image_height = 640, 480
    fx, fy, cx, cy = 500.0, 500.0, image_width / 2, image_height / 2
    camera_model = cameras.PinholeCamera(
        K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        image_width=image_width,
        image_height=image_height,
    )

    points_image = np.tile(np.array([cx, cy]), (6, 5, 4, 1))
    points_camera = camera_model.unproject(points_image)
    self.assertEqual(points_camera.shape, (6, 5, 4, 3))
    self.assertAllClose(
        points_camera, np.tile(np.array([0.0, 0.0, 1.0]), (6, 5, 4, 1))
    )

  def test_unproject_points_to_ray_directions(self):
    """Unproject a grid of 2D image points as rays and verify ray properties."""
    num_points = 10
    image_width, image_height = 640, 480
    points_image = np.column_stack(
        (
            np.random.uniform(0.0, image_width, num_points),
            np.random.uniform(0.0, image_height, num_points),
        )
    )
    intrinsics = np.array(
        [[500, 0.0, 320.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]]
    )
    camera_model = cameras.PinholeCamera(
        K=intrinsics, image_width=image_width, image_height=image_height
    )
    points_camera = camera_model.unproject(points_image, to_rays=True)
    self.assertEqual(points_camera.shape, (num_points, 3))

    # Assert thet each ray is a unit vector.
    self.assertAllClose(tf.norm(points_camera, axis=-1), tf.ones(num_points))

    # Assert thet each ray project backs to the same image coordinate.
    for point_camera, point_image in zip(points_camera, points_image):
      expected = intrinsics.dot(point_camera)
      expected = expected[:2] / expected[2]
      self.assertAllClose(point_image, expected)

  def test_unproject_points_to_unit_zplane(self):
    """Unproject a grid of 2D image points to z=1 plane and verify correctness.
    """
    num_points = 10
    image_width, image_height = 640, 480
    points_image = np.column_stack(
        (
            np.random.uniform(0.0, image_width, num_points),
            np.random.uniform(0.0, image_height, num_points),
        )
    )
    intrinsics = np.array(
        [[500, 0.0, 320.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]]
    )
    camera_model = cameras.PinholeCamera(
        K=intrinsics, image_width=image_width, image_height=image_height
    )
    points_camera = camera_model.unproject(points_image, to_rays=False)
    self.assertEqual(points_camera.shape, (num_points, 3))

    # Assert thet each ray end point is at z=1.
    self.assertAllClose(points_camera[:, 2], tf.ones(num_points))

    # Assert thet each ray project backs to the same image coordinate.
    for point_camera, point_image in zip(points_camera, points_image):
      expected = intrinsics.dot(point_camera)
      expected = expected[:2] / expected[2]
      self.assertAllClose(point_image, expected)

  def test_unproject_and_project(self):
    """Unproject image points to 3D and then project back to verify they are same.
    """
    # Create an pinhole camera.
    camera_model = cameras.PinholeCamera.from_intrinsics(
        image_size_in_pixels=(320, 240), focal_length_in_pixels=(400.0, 400.0)
    )

    # Calling unproject without argument unprojects all pixel centers.
    ray_directions = camera_model.unproject()

    # Re-project the ray directions to verify they are same.
    projections = camera_model.project(ray_directions)
    self.assertAllClose(projections, camera_model.pixel_centers(), atol=1e-3)

  def test_pixel_centers(self):
    """Verifies the correctness of `pixel_centers` with an alternate implementation.
    """
    # Create an pinhole camera.
    image_width, image_height = 320, 240
    camera_model = cameras.PinholeCamera.from_intrinsics(
        image_size_in_pixels=(image_width, image_height),
        focal_length_in_pixels=(400.0, 400.0),
    )

    # Create an array (H, W, 2) containing pixel center coordinates.
    pixel_centers = tf.meshgrid(
        tf.linspace(0.5, image_width - 0.5, image_width),
        tf.linspace(0.5, image_height - 0.5, image_height),
        indexing='xy',
    )
    pixel_centers = tf.stack(pixel_centers, axis=-1)

    # Make sure the pixel centers are correct.
    self.assertAllClose(camera_model.pixel_centers(), pixel_centers)
