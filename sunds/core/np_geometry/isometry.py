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

"""Geometric transforms (e.g. rigid transformation)."""

import dataclasses
from typing import Union

import numpy as np


@dataclasses.dataclass
class Isometry:
  """3D transform object used to represent an SE(3) (isometric) transform.

  Underneath this class stores the transform [R|t] composed of rotation (R) and
  translation (t).

  Usage example:

  ```python
  frameB_from_frameA = Isometry(R=np.eye(3), t=np.ones(3))
  pointA = np.random.rand(3)
  pointB = frameB_from_frameA * pointA
  pointA = frameB_from_frameA.inverse() * pointB

  # Compose multiple transforms:
  frameA_to_frameB = Isometry(...)
  frameB_to_frameC = Isometry(...)
  frameA_to_frameC = frameB_to_frameC * frameA_to_frameB

  # Apply transform on single point:
  pointB = frameA_to_frameB * np.array([4.0, 2.0, 1.0])

  # Apply transform on a pointcloud (Nx3):
  pointcloudC = frameA_to_frameC * np.random.rand(1000, 3)
  ```

  """

  # Rotation component with tensor shape (3, 3)
  R: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))  # pylint: disable=invalid-name
  # Translation component with tensor shape (3,)
  t: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))

  @classmethod
  def from_matrix(cls, matrix: np.ndarray) -> 'Isometry':
    """Constructs from a 3x4 or 4x4 transform matrix."""
    if matrix.shape not in [(3, 4), (4, 4)]:
      raise ValueError('invalid matrix.shape={}'.format(matrix.shape))
    return cls(R=matrix[:3, :3], t=matrix[:3, 3])

  def matrix3x4(self) -> np.ndarray:
    """Returns as 3x4 matrix.

    Returns a matrix [R|t] of shape (3, 4)
    """
    return np.hstack((self.R, self.t.reshape((3, 1))))

  def matrix4x4(self) -> np.ndarray:
    """Returns as 4x4 matrix.

    Returns a matrix [R|t] of shape (4, 4)
                     [0|1]
    """
    matrix = np.eye(4)
    matrix[:3, :3] = self.R
    matrix[:3, 3] = self.t
    return matrix

  def inverse(self) -> 'Isometry':
    """Returns the inverse of self.

    Usage example:

    frameB_from_frameA = Isometry(R=np.eye(3), t=np.ones(3))
    frameA_from_frameB = frameB_from_frameA.inverse()

    Returns:
      Inverse transform of self.
    """
    return Isometry(self.R.T, -self.R.T.dot(self.t))

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
    return Isometry(self.R.dot(other.R), self.R.dot(other.t) + self.t)

  def transform_points(self, points: np.ndarray) -> np.ndarray:
    """Computes the transformation of a set of points.

    frameA_to_frameB = Isometry()
    pointsA = np.random.rand(1000, 3)
    pointsB = frameA_to_frameB.transform_points(pointsA)

    Args:
      points: Tensor containing point positions of shape (N, 3) or a single
        point vector of shape (3,).

    Returns:
      Transformed points.
    """
    projected = np.einsum('ij,nj->ni', self.R, points.reshape(-1, 3)) + self.t
    return np.squeeze(projected)

  def __mul__(
      self, other: Union['Isometry',
                         np.ndarray]) -> Union['Isometry', np.ndarray]:
    """Returns the product of self with other i.e. `out = self * other`.

    This function can be used to transform point(s) or compose multiple
    transforms together.

    Compose multiple transforms:
    frameA_to_frameB = Isometry(...)
    frameB_to_frameC = Isometry(...)
    frameA_to_frameC = frameB_to_frameC * frameA_to_frameB

    Apply transform on single point:
    pointB = frameA_to_frameB * np.array([4.0, 2.0, 1.0])

    Apply transform on a pointcloud (Nx3):
    pointcloudC = frameA_to_frameC * np.random.rand(1000, 3)

    Args:
      other: Either 3D point(s) or vector(s) to transform or other transform to
        compose with.

    Returns:
      When multiplying with another Isometry object `other`, the composite
      transform equal to `(this * other)` is returned. When other is point with
      shape (3,) or a pointcloud of shape (N, 3), the output is the transformed
      point or pointcloud.
    """
    if isinstance(other, np.ndarray):
      return self.transform_points(other)
    elif isinstance(other, Isometry):
      return self.compose(other)
    raise TypeError('Unsupported type')
