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

"""Camera module."""

import abc
import dataclasses
from typing import TypeVar, Optional, Tuple

import tensorflow as tf

TensorLike = tf.types.experimental.TensorLike

# Type annotation of any instance of subclass of `AbstractCamera`.
CameraType = TypeVar('CameraType', bound='AbstractCamera')


@dataclasses.dataclass
class AbstractCamera(abc.ABC):
  """An abstract camera class that defines the interface of camera models.

  This class provides abstract interface expected by different camera model
  imaplementations. Custom camera model classes are expected to subclass from
  `AbstractCamera` and implement the abstract methods. See `PinholeCamera` for
  an example of a concrete class subclassing `AbstractCamera`.
  """

  @abc.abstractmethod
  def project(self, points: TensorLike) -> tf.Tensor:
    """Project 3D points in camera frame to image frame.

    Given a set of 3D points in camera frame, this method returns the 2D image
    projections.

    Note that extrinsics is currently not part of the camera model and thus this
    `project` method expects the input 3D points to be in camera space. So world
    space points should be first transformed to camera space before passing them
    to this method.

    This method supports arbritrary batching of input points tensor (including
    no batching in case of single 3D point as input).

    This abstract method should be implemented in the subclassed camera model.

    Args:
      points: A tensor of shape `(..., 3)` containing 3D point positions in the
        camera frame.

    Returns:
      2D image projections in a `(..., 2)` tensor.
    """
    pass

  @abc.abstractmethod
  def unproject(self,
                points: Optional[TensorLike] = None,
                to_rays: bool = True) -> tf.Tensor:
    """Unproject 2D pixel points in image space to camera space.

    Given a a set of 2d point coordinates in the camera image, this function
    unprojects each 2D point to either a 3D ray or point in camera space. When
    `to_rays` is set to `True` (default), the output will be normalized 3D ray
    direction vectors in camera frame passing through every 2D image point.
    When `to_rays` is set to `False`, the output will be 3D points along the
    rays at `z=1` plane.

    This function supports arbritrary batching of input points tensor (including
    no batching for a single point).

    This abstract method should be implemented in the subclassed camera model.

    Args:
      points: A tensor of shape `(..., 2)` containing 2D image projections. If
        not provided, all pixel centers will be unprojected.
      to_rays: If `True` the output will normalized ray direction vectors in the
        camera space passing through every 2D image point. Otherwise the output
        will be 3D points along the rays in camera frame at `z=1` plane.

    Returns:
      A tensor of shape `(..., 3)` containing the un-projected ray directions
      (or points along rays at `z=1`) passing though each input 2D image points.
    """
    pass

  @abc.abstractmethod
  def pixel_centers(self) -> tf.Tensor:
    """Returns 2D coordinates of centers of all pixels in the camera image.

    The pixel centers of camera image are returned as a float32 tensor of shape
    `(image_height, image_width, 2)`.

    This abstract method should be implemented in the subclassed camera model.

    Returns:
      2D image coordinates of center of all pixels of the camer image in tensor
      of shape `(image_height, image_width, 2)`.
    """
    pass


@dataclasses.dataclass
class PinholeCamera(AbstractCamera):
  """Linear (pin-hole) camera model.

  A linear camera model, where intrinsics is 3 x 3 matrix e.g.
      [fx  s  cx]
  K = [0  fy  cy]
      [0   0   1]
  where fx, fy being focal length in pixels, s being skew, and cx, cy is
  principal point in pixels.

  This camera model uses the convention that top left corner of the image is
  `(0, 0)` and bottom right corner is `(image_width, image_height)`. So the
  center of the top left corner pixel is (0.5, 0.5).

  Example Usage:

  ```python
  # Create a pinhole camera instance with default constructor.
  camera = isun.geometry.PinholeCamera(
      K=[[fx, s, cx], [0, fy, cy], [0, 0, 1]],
      image_width=W,
      image_height=H,
  )

  # Alternatively, it is also possible to construct a pinhole camera instance
  # using classmethods like `from_fov` and `from_intrinsics`.

  # Construct a pinhole camera with horizontal field of view of 45°.
  camera = isun.geometry.PinholeCamera.from_fov(
      image_width=W,
      image_height=H,
      horizontal_fov_in_degrees=45.0)

  # Construct a pinhole camera from common intrinsics parameters.
  camera = isun.geometry.PinholeCamera.from_intrinsics(
      image_size_in_pixels=(W, H),
      focal_length_in_pixels=(fx, fy))

  # Project and unproject 1000 random points.
  image_points = camera.project(np.random.rand(1000, 3))
  camera_rays = camera.unproject(image_points)

  # Project and unproject single point.
  image_point = camera.project(tf.constant([320., 240.]))
  camera_rays = camera.unproject(image_point)

  # Create an array (H, W, 2) containing pixel center coordinates.
  pixel_centers = camera.pixel_centers()

  # Generate a pointcloud from depth image storing per pixel depth along rays.
  pointcloud = camera.unproject(pixel_centers, to_rays=True) * depth

  # Generate a pointcloud from depth image storing per pixel depth along Z.
  pointcloud = camera.unproject(pixel_centers, to_rays=False) * depth
  ```
  """
  # 3x3 camera intrinsics matrix.
  K: TensorLike  # pylint: disable=invalid-name
  # Width of the camera image in pixels.
  image_width: int = 0
  # Height of the camera image in pixels.
  image_height: int = 0

  def __post_init__(self):
    self.K = tf.convert_to_tensor(self.K)
    self.K = tf.ensure_shape(self.K, (3, 3))

  @classmethod
  def from_fov(
      cls,
      *,
      image_width: int,
      image_height: int,
      horizontal_fov_in_degrees: float,
      vertical_fov_in_degrees: Optional[float] = None) -> 'PinholeCamera':
    """Creates a `PinholeCamera` from field of view parameters.

    This `classmethod` provides a convenient way to construct an `PinholeCamera`
    instance with common field of view (fov) parameters.

    Example Usage:

    ```python
    # Construct a pinhole camera with horizontal field of view of 45°.
    camera = isun.geometry.PinholeCamera.from_fov(
        image_width=W,
        image_height=H,
        horizontal_fov_in_degrees=45.0)
    ```

    Args:
      image_width: Camera image width in pixels.
      image_height: Camera image height in pixels.
      horizontal_fov_in_degrees: Horizontal fov of the camera in degrees.
      vertical_fov_in_degrees: Optional vertical fov of the camera in degrees.
        When set to `None` (default) we assume square pixel aspect ratio, and
        vertical focal length will be set to same as horizontal focal length.

    Returns:
      A `PinholeCamera` instance with provided intrinsics.
    """

    def focal_length_from_fov(image_size, fov_in_degrees):
      fov_in_radians = tf.experimental.numpy.deg2rad(fov_in_degrees)
      focal_length = .5 * image_size / tf.math.tan(.5 * fov_in_radians)
      return tf.cast(focal_length, tf.float32)

    fx = focal_length_from_fov(image_width, horizontal_fov_in_degrees)
    fy = fx if vertical_fov_in_degrees is None else focal_length_from_fov(
        image_height, vertical_fov_in_degrees)

    cx, cy = image_width / 2, image_height / 2
    return cls(
        image_width=image_width,
        image_height=image_height,
        K=tf.convert_to_tensor([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]))

  @classmethod
  def from_intrinsics(
      cls,
      *,
      image_size_in_pixels: Tuple[int, int],
      focal_length_in_pixels: Tuple[float, float],
      principal_point_in_pixels: Optional[Tuple[float, float]] = None,
      skew: float = 0.0,
  ) -> 'PinholeCamera':
    """Creates a `PinholeCamera` from intrinsic parameters.

    This `classmethod` provides a convenient way to construct an `PinholeCamera`
    instance with common intrinsic parametrs.

    Example Usage:

    ```python
    # Construct a pinhole camera from common intrinsics parameters.
    camera = isun.geometry.PinholeCamera.from_intrinsics(
        image_size_in_pixels=(W, H),
        focal_length_in_pixels=(fx, fy),
        principal_point_in_pixels=(cx, cy),
        skew=skew)
    ```

    Args:
      image_size_in_pixels: (width, height) of the camera image in pixels.
      focal_length_in_pixels: (horizontal, vertical) focal length in pixels.
      principal_point_in_pixels: Optional (horizontal, vertical) principal point
        in pixels. If set to `None` (default), principal point is image center.
      skew: Skew coefficient.

    Returns:
      A `PinholeCamera` instance with provided intrinsics.
    """
    image_width, image_height = image_size_in_pixels
    fx, fy = focal_length_in_pixels
    if principal_point_in_pixels is None:
      principal_point_in_pixels = (image_width / 2, image_height / 2)
    cx, cy = principal_point_in_pixels
    return cls(
        image_width=image_width,
        image_height=image_height,
        K=tf.convert_to_tensor([[fx, skew, cx], [0., fy, cy], [0., 0., 1.]]))

  def project(self, points: TensorLike) -> tf.Tensor:
    """Project 3D points in camera frame to image frame.

    Given a set of 3D points in camera frame, this method returns the 2D image
    projections.

    Note that extrinsics is currently not part of the camera model and thus this
    `project` method expects the input 3D points to be in camera space. So world
    space points should be first transformed to camera space before passing them
    to this method.

    This method supports arbritrary batching of input points tensor (including
    no batching in case of single 3D point as input).

    Args:
      points: A tensor of shape `(..., 3)` containing 3D point positions in the
        camera frame.

    Returns:
      2D image projections in a `(..., 2)` tensor.
    """
    image_frame = tf.einsum('ij,...j->...i', self.K, points)
    image_frame = (image_frame[..., :2] / image_frame[..., 2:3])
    return image_frame

  def unproject(self,
                points: Optional[TensorLike] = None,
                to_rays: bool = True) -> tf.Tensor:
    """Unproject 2D pixel points in image space to camera space.

    Given a a set of 2d point coordinates in the camera image, this functions
    unprojects each 2D point to either a 3D ray or point in camera space. When
    `to_rays` is set to `True` (default), the output will normalized 3D ray
    direction vectors in camera frame passing through every 2D image point.
    When `to_rays` is set to `False`, the output will be 3D points along the
    rays at `z=1` plane.

    This function supports arbritrary batching of input points tensor (including
    no batching for a single point).

    Args:
      points: A tensor of shape `(..., 2)` containing 2D image projections. If
        not provided, all pixel centers will be unprojected.
      to_rays: If `True` the output will normalized ray direction vectors in the
        camera space passing through every 2D image point. Otherwise the output
        will be 3D points along the rays in camera frame at `z=1` plane.

    Returns:
      A tensor of shape `(..., 3)` containing the un-projected ray directions
      (or points along rays at `z=1`) passing though each input 2D image points.
    """
    if points is None:
      points = self.pixel_centers()

    image_frame = tf.concat((points, tf.ones_like(points[..., 0:1])), axis=-1)
    camera_frame = tf.einsum('ij,...j->...i', tf.linalg.inv(self.K),
                             image_frame)
    if to_rays:
      camera_frame, _ = tf.linalg.normalize(camera_frame, axis=-1)
    else:
      camera_frame = camera_frame / tf.expand_dims(
          camera_frame[..., 2], axis=-1)
    return camera_frame

  def pixel_centers(self) -> tf.Tensor:
    """Returns 2D coordinates of centers of all pixels in the camera image.

    The pixel centers of camera image are returned as a float32 tensor of shape
    `(image_height, image_width, 2)`.

    This camera model uses the convention that top left corner of the image is
    `(0, 0)` and bottom right corner is `(image_width, image_height)`. So the
    center of the top left corner pixel is `(0.5, 0.5)`.

    Returns:
      2D image coordinates of center of all pixels of the camer image in tensor
      of shape `(image_height, image_width, 2)`.
    """
    image_grid = tf.meshgrid(
        tf.range(self.image_width), tf.range(self.image_height), indexing='xy')
    return tf.cast(tf.stack(image_grid, axis=-1), dtype=tf.float32) + 0.5
