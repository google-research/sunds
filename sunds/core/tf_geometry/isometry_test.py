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

"""Tests for isun.geometry.transforms."""

import numpy as np
from sunds.core import tf_geometry
import tensorflow as tf


def _rotation_around_x(angle: float) -> np.ndarray:
  """Rotation matrix for rotation around X."""
  c = np.cos(angle)
  s = np.sin(angle)
  rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
  return rot_x


class IsometryTest(tf.test.TestCase):

  def test_identity_transform_as_matrix3x4(self):
    """Verifies the `matrix3x4` method for identity transform."""
    tfm = tf_geometry.Isometry(R=tf.eye(3), t=tf.zeros(3))
    expected = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    self.assertAllClose(tfm.matrix3x4(), expected)

  def test_identity_transform_as_matrix4x4(self):
    """Verifies the `matrix4x4` method for identity transform."""
    tfm = tf_geometry.Isometry(R=tf.eye(3), t=tf.zeros(3))
    self.assertAllClose(tfm.matrix4x4(), tf.eye(4))

  def test_identity_transform_inverse(self):
    """Verifies that the inverse of identity transform is identity."""
    tfm = tf_geometry.Isometry(R=tf.eye(3), t=tf.zeros(3))
    self.assertAllClose(tfm.matrix3x4(), tfm.inverse().matrix3x4())

  def test_from_matrix(self):
    """Construct from matrix and verify attributes."""
    tfm = tf_geometry.Isometry.from_matrix(tf.eye(4))
    self.assertAllClose(tfm.R, np.eye(3))
    self.assertAllClose(tfm.t, np.zeros(3))

  def test_transform_composition(self):
    """Tests that composition of transform with its inverse is identity."""
    tfm = tf_geometry.Isometry(R=tf.eye(3), t=tf.constant([1., 0., 0.]))
    tfm_product = tfm * tfm.inverse()
    self.assertIsInstance(tfm_product, tf_geometry.Isometry)
    self.assertAllClose(tfm_product.R, tf.eye(3))
    self.assertAllClose(tfm_product.t, tf.zeros(3))

  def test_point_transform(self):
    """Verifies correctness of transforming a single 3D points."""
    rotation = _rotation_around_x(np.pi / 2)
    translation = np.array([1., 0., 0.])
    tfm = tf_geometry.Isometry(rotation, translation)
    for _ in range(10):
      point = np.random.rand(3)
      projected = tfm * point
      self.assertIsInstance(projected, tf.Tensor)
      self.assertAllClose(projected, rotation.dot(point) + translation)

  def test_pointcloud_transform(self):
    """Verifies correctness of transforming a single 3D point cloud."""
    rotation = _rotation_around_x(np.pi / 2)
    translation = np.array([1., 0., 0.])
    earth_to_mars = tf_geometry.Isometry(rotation, translation)
    points_earth = np.random.rand(10, 3)
    points_mars = earth_to_mars * points_earth
    for point_earth, point_mars in zip(points_earth, points_mars):
      self.assertAllClose(point_mars, rotation.dot(point_earth) + translation)

  def test_batched_pointcloud_transform(self):
    """Verifies correctness of transforming a multiple 3D point clouds."""
    rotation = _rotation_around_x(np.pi / 2)
    translation = np.array([1., 0., 0.])
    earth_to_mars = tf_geometry.Isometry(rotation, translation)
    clouds_earth = np.random.rand(2, 10, 3)
    clouds_mars = earth_to_mars * tf.convert_to_tensor(clouds_earth)
    for points_earth, points_mars in zip(clouds_earth, clouds_mars):
      for point_earth, point_mars in zip(points_earth, points_mars):
        self.assertAllClose(point_mars, rotation.dot(point_earth) + translation)
