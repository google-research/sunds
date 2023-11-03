# Copyright 2023 The sunds Authors.
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

"""Tests for transform."""

from absl.testing import parameterized
import numpy as np
from sunds.core.np_geometry.isometry import Isometry


def _rotation_around_x(angle: float) -> np.ndarray:
  """Rotation matrix for rotation around X."""
  c = np.cos(angle)
  s = np.sin(angle)
  rot_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
  return rot_x


class IsometryTest(parameterized.TestCase):

  def test_pose_init_default(self):
    pose = Isometry()
    np.testing.assert_equal(pose.R, np.eye(3))
    np.testing.assert_equal(pose.t, np.zeros(3))

    expected = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    np.testing.assert_equal(pose.matrix3x4(), expected)
    np.testing.assert_array_almost_equal(pose.inverse().matrix3x4(), expected)

  def test_pose_init_only_translation(self):
    pose = Isometry(t=np.array([1.0, 0.0, 0.0]))

    mat = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    np.testing.assert_equal(pose.matrix3x4(), mat)

    mat_inv = np.array(
        [[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    np.testing.assert_array_almost_equal(pose.inverse().matrix3x4(), mat_inv)

  def test_pose_from_matrix(self):
    matrix = np.eye(4)
    pose = Isometry.from_matrix(matrix)
    np.testing.assert_equal(pose.R, np.eye(3))
    np.testing.assert_equal(pose.t, np.zeros(3))
    np.testing.assert_equal(pose.matrix4x4(), matrix)

  @parameterized.named_parameters(
      ('=5x5', np.eye(5)),
      ('=2x2', np.eye(2)),
  )
  def test_pose_from_invalid_matrix(self, matrix):
    with self.assertRaises(ValueError):
      _ = Isometry.from_matrix(matrix)

  def test_pose_times_pose(self):
    pose = Isometry(np.eye(3), np.array([1.0, 0.0, 0.0]))
    pose_product = pose * pose.inverse()
    self.assertIsInstance(pose_product, Isometry)
    np.testing.assert_equal(pose_product.matrix3x4(), Isometry().matrix3x4())

  def test_pose_times_vector(self):
    rotation = _rotation_around_x(np.pi / 2)
    translation = np.array([1.0, 0.0, 0.0])
    pose = Isometry(rotation, translation)
    for _ in range(10):
      point = np.random.rand(3)
      projected = pose * point
      self.assertIsInstance(projected, np.ndarray)
      np.testing.assert_equal(projected, rotation.dot(point) + translation)

  def test_pose_times_matrix(self):
    rotation = _rotation_around_x(np.pi / 2)
    translation = np.array([1.0, 0.0, 0.0])
    earth_to_mars = Isometry(rotation, translation)
    points_earth = np.random.rand(10, 3)
    points_mars = earth_to_mars * points_earth
    for point_earth, point_mars in zip(points_earth, points_mars):
      np.testing.assert_equal(
          point_mars, rotation.dot(point_earth) + translation
      )

  @parameterized.named_parameters(
      ('=invalid_tensor_ndim', np.random.rand(1, 1, 1)),
      ('=invalid_tensor_shape', np.random.rand(10, 5)),
  )
  def test_pose_times_invalid_tensor_shape(self, other):
    with self.assertRaises(ValueError):
      _ = Isometry() * other
