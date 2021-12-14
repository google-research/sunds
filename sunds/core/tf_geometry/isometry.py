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

"""Geometric transforms (e.g. rigid transformation)."""

import dataclasses
from typing import Union

import tensorflow as tf

TensorLike = tf.types.experimental.TensorLike


@dataclasses.dataclass
class Isometry:
  """3D transform object used to represent an SE(3) (isometric) transform.

  Underneath this class stores the transform [R|t] composed of rotation (R) and
  translation (t).

  Usage example:

  frameB_from_frameA = Isometry(R=tf.eye(3), t=tf.ones(3))
  pointA = tf.random.uniform(shape=(3,))
  pointB = frameB_from_frameA * pointA
  pointA = frameB_from_frameA.inverse() * pointB

  Compose multiple transforms:
  frameA_to_frameB = Isometry(...)
  frameB_to_frameC = Isometry(...)
  frameA_to_frameC = frameB_to_frameC * frameA_to_frameB

  Apply transform on single point:
  pointB = frameA_to_frameB * tf.constant([4.0, 2.0, 1.0])

  Apply transform on a pointcloud (Nx3):
  pointcloudC = frameA_to_frameC * tf.random.uniform(shape=(1000, 3))

  Apply transform on multiple batches of pointcloud (MxNx3):
  pointcloudC = frameA_to_frameC * tf.random.uniform(shape=(5, 1000, 3))
  """

  # Rotation component with tensor shape (3, 3)
  R: TensorLike  # pylint: disable=invalid-name
  # Translation component with tensor shape (3,)
  t: TensorLike

  def __post_init__(self):
    self.R = tf.ensure_shape(tf.convert_to_tensor(self.R), (3, 3))
    self.t = tf.ensure_shape(tf.convert_to_tensor(self.t), (3,))

  @classmethod
  def from_matrix(cls, matrix: TensorLike) -> 'Isometry':
    """Constructs from a 3x4 or 4x4 transform matrix."""
    return cls(R=matrix[:3, :3], t=matrix[:3, 3])

  def matrix3x4(self) -> tf.Tensor:
    """Returns as 3x4 matrix.

    Returns a matrix [R|t] of shape (3, 4)
    """
    return tf.concat((self.R, tf.reshape(self.t, (3, 1))), axis=1)

  def matrix4x4(self) -> tf.Tensor:
    """Returns as 4x4 matrix.

    Returns a matrix [R|t] of shape (4, 4)
                     [0|1]
    """
    return tf.concat((self.matrix3x4(), [[0, 0, 0, 1]]), axis=0)

  def inverse(self) -> 'Isometry':
    """Returns the inverse of self.

    Usage example:

    frameB_from_frameA = Isometry(R=tf.eye(3), t=tf.ones(3))
    frameA_from_frameB = frameB_from_frameA.inverse()

    Returns:
      Inverse transform of self.
    """
    return Isometry(
        R=tf.transpose(self.R),
        t=-tf.linalg.matvec(tf.transpose(self.R), self.t))

  def compose(self, other: 'Isometry') -> 'Isometry':
    """Returns the composite transform equal to self * other.

    This function is used to compose multiple transforms together. This can
    alternatively be achieved via `*` operator.

    Usage example:

    frameB_from_frameA = Isometry(R=..., t=...)
    frameC_from_frameB = Isometry(R=..., t=...)

    frameC_from_frameA = frameC_from_frameB.compose(frameB_from_frameA)

    Args:
      other: Another transform to compose with.

    Returns:
      Composite transform equal to self * other.
    """
    return Isometry(
        R=self.R @ other.R, t=tf.linalg.matvec(self.R, other.t) + self.t)

  def transform_points(self, points: TensorLike) -> tf.Tensor:
    """Computes the transformation of a set of points.

    frameA_to_frameB = Isometry()
    pointsA = tf.random.uniform(shape=(1000, 3))
    pointsB = frameA_to_frameB.transform_points(pointsA)

    Args:
      points: Tensor containing point positions of shape (..., 3).

    Returns:
      Transformed points.
    """
    return tf.einsum('ij,...j->...i', self.R, points) + self.t

  def __mul__(
      self, other: Union['Isometry',
                         TensorLike]) -> Union['Isometry', tf.Tensor]:
    """Returns the product of self with other i.e.

    out = self * other.

    This function can be used to transform point(s) or compose multiple
    transforms together.

    Compose multiple transforms:
    frameA_to_frameB = Isometry(...)
    frameB_to_frameC = Isometry(...)
    frameA_to_frameC = frameB_to_frameC * frameA_to_frameB

    Apply transform on single point:
    pointB = frameA_to_frameB * tf.constant([4.0, 2.0, 1.0])

    Apply transform on a pointcloud (Nx3):
    pointcloudC = frameA_to_frameC * tf.random.uniform(shape=(1000, 3))

    Apply transform on multiple batches of pointcloud (MxNx3):
    pointcloudC = frameA_to_frameC * tf.random.uniform(shape=(5, 1000, 3))

    Args:
      other: Either 3D point(s) to transform or other transform to compose with.

    Returns:
      When multiplying with another Isometry object `other`, the composite
      transform equal to `(this * other)` is returned. When `other` is a point
      cloud tensor with shape (..., 3), the output is the transformed point
      cloud (with same shape).
    """
    if isinstance(other, Isometry):
      return self.compose(other)
    else:
      return self.transform_points(other)
