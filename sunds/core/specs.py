# Copyright 2021 The Sunds Authors.
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

"""Data specification of features of scene understanding datasets."""

from typing import Tuple

from sunds.core import spec_dict
from sunds.typing import Dim, FeatureSpecs, FeaturesOrBool, LabelOrFeaturesOrBool  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


def scene_spec(
    frames: FeatureSpecs,  # pylint: disable=redefined-outer-name
    *,
    point_cloud: FeaturesOrBool = False,
) -> FeatureSpecs:
  """Scene spec definitions."""
  frames = dict(frames)
  frames.pop('scene_name')  # Clear the scene field
  specs = spec_dict.SpecDict({
      # A unique name 🔑 that identifies the scene.
      'scene_name': tfds.features.Text(),
      # A scene has several frames. This stores a lightweight information of all
      # frames (without sensor data) in the scene. This can be used for random
      # lookup of a particular frame from the `frame` store.
      'frames': tfds.features.Sequence(frames),
      # Scene bounding box 📦. This axis aligned bounding box can be used to
      # represent the extent of the scene in its local coordinate frame.
      'scene_box': aligned_box_3d_spec(),
      # Nominal time ⏰ of the scene encoded as `RFC3339_full` format datetime
      # string e.g. `1970-01-01 00:00:00.0 +0000`. All timestamps in `frame`
      # level data are expected to be relative (elapsed time in seconds) to this
      # nominal time. This field can be left unspecified for most datasets
      # unless there is a need to explicitly get the absolute time point.
      'nominal_time': tfds.features.Text(),
  })
  # LiDAR point cloud.
  specs.maybe_set('point_cloud', point_cloud)
  return specs


def frame_spec(cameras: FeatureSpecs) -> FeatureSpecs:
  """Frame specification used for storing frame data in `frame` stores.

  A `frame` is typically used to group sensor measurements taken during a small
  timespan with a shared sensor rig or rosette. For example `frame` may be used
  to group all cameras and lidar sensor information and observations captured at
  a particular timespan during a run of an autonomous driving vehicle. So each
  frame can have multiple cameras (e.g. stereo setups) and lidars ( e.g.
  autonomous cars with multiple lidar sensors). However in this simple synthetic
  dataset, each frame has only one camera.

  See `frame_info_spec()` for a lightweight version (without sensor data)  of
  `frame` which is used to store information about all frames in a scene.

  Args:
    cameras: A dict[camera_name, sunds.specs.camera_spec()]

  Returns:
    A composite `tfds` feature defining the specification of `frame` data.
  """
  return {
      # A unique name 🔑 that identifies the sequence the frame is from.
      'scene_name': tfds.features.Text(),
      # A unique name 🔑 that identifies this particular frame.
      'frame_name': tfds.features.Text(),
      # Frame pose w.r.t scene: X_{scene} = R * X_{frame} + t.
      'pose': pose_spec(),
      # Frame timestamp ⏰. This is expected to be the timestamp when frame
      # `pose` was recorded. We expect this timestamp to be relative to the
      # nominal timestamp of the scene this frame belongs to.
      'timestamp': tf.float32,
      # Camera sensor data. Each frame can have multiple cameras (e.g. stereo
      # setups, autonomous cars with multiple cameras). See `camera_spec` for
      # more details about the contents of each camera.
      'cameras': cameras,
  }


def camera_spec(
    *,
    color_image: FeaturesOrBool = False,
    category_image: LabelOrFeaturesOrBool = False,
    depth_image: FeaturesOrBool = False,
    camera_rays: FeaturesOrBool = False,
    img_shape: Tuple[Dim, Dim] = (None, None),
) -> FeatureSpecs:
  """Feature specification of camera sensor 📷.

  This functions returns the specification of camera sensor data like intrinsics
  and extrinsics of the camera, and optionally the images caputured by the
  camera and image level annotations.

  Note that the camera extrinsics stored here are w.r.t frame. To get the pose
  of a camera w.r.t to scene, we have to also use the pose of the frame w.r.t
  scene.

  Args:
    color_image: Rgb color image is stored.
    category_image: Category segmentation label image.
    depth_image: depth image is stored.
    camera_rays: The given camera specs.
    img_shape: The (h, w) image shape

  Returns:
    A composite `tfds` feature defining the specification of camera data.
  """
  spec = spec_dict.SpecDict({
      # Camera intrinsics.
      'intrinsics': camera_intrinsics_spec(),
      # Camera extrinsics w.r.t frame (frame to camera transform):
      # X_{camera} = R * X_{frame} + t.
      # If a camera is not posed, this can be left to `Identity`.
      'extrinsics': pose_spec(),
  })
  # Color image data.
  spec.maybe_set(
      'color_image',
      color_image,
      tfds.features.Image(shape=(*img_shape, 3)),
  )
  # Category segmentation data.
  spec.maybe_set(
      'category_image',
      category_image,
      spec_dict.labeled_image(shape=(*img_shape, 1)),
  )
  # Depth image.
  spec.maybe_set(
      'depth_image',
      depth_image,
      tfds.features.Image(shape=(*img_shape, 1), dtype=tf.float32),
  )
  # Camera rays
  spec.maybe_update(
      camera_rays,
      camera_rays_spec(img_shape=img_shape),
  )
  return spec


def camera_intrinsics_spec() -> FeatureSpecs:
  """Specification of camera intrinsics.

  The camera instrisics model is identical to the `opencv` and `vision::sfm`
  camera calibration model. This is used in `camera_spec` which has other camera
  data like extrinsics of the camera and image data.

  Returns:
    A composite `tfds` feature defining the specification of camera intrinsics.
  """
  return {
      # Image width of the camera sensor.
      'image_width': tf.int32,
      # Image height of the camera sensor.
      'image_height': tf.int32,
      # Camera intrinsics matrix K (3x3 matrix).
      #     [fx skew cx]
      # K = [ O   fy cy]
      #     [ 0    0  1]
      'K': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
      # Camera projection type. Should be either 'PERSPECTIVE' | 'FISHEYE'. For
      # the `nerf_synthetic` data this is always `PERSPECTIVE` (pinhole).
      'type': tfds.features.Text(),
      # Camera distortion coefficients. Since cameras in this dataset does not
      # have any distortions, these will have zero values and can be ignored.
      'distortion': {
          # Radial distortion coefficients [k1, k2, k3].
          'radial': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
          # Tangential distortion coefficients [p1, p2].
          'tangential': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
      }
  }


def pose_spec() -> FeatureSpecs:
  """Specification of pose represented by 3D Isometric transformation.

  Returns:
    A composite `tfds` feature defining the specification of SE(3) pose data.
  """
  return {
      # 3x3 rotation matrix.
      'R': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
      # 3D translation vector.
      't': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
  }


def camera_rays_spec(
    *,
    img_shape: Tuple[Dim, Dim] = (None, None),
    encoding: tfds.features.Encoding = tfds.features.Encoding.ZLIB,
) -> FeatureSpecs:
  """Specification for explicit camera rays."""
  return {
      'ray_directions':
          tfds.features.Tensor(
              shape=(*img_shape, 3),
              dtype=tf.float32,
              encoding=encoding,
          ),
      'ray_origins':
          tfds.features.Tensor(
              shape=(*img_shape, 3),
              dtype=tf.float32,
              encoding=encoding,
          ),
  }


def aligned_box_3d_spec() -> tfds.features.FeaturesDict:
  """Specification of an Axis aligned bounding box 📦."""
  return {
      # A box is considered null (empty) if any(min > max).
      # Minimum extent of an axis aligned box.
      'min_corner': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
      # Maximum extent of an axis aligned box.
      'max_corner': tfds.features.Tensor(shape=(3,), dtype=tf.float32),  # pytype: disable=bad-return-type  # gen-stub-imports
  }


def point_cloud_spec(
    *,
    category_labels: FeaturesOrBool = False,
) -> FeatureSpecs:
  """Specification of a LiDAR point cloud."""
  # TODO(epot): Rather than using "None" for the first dimension of each Tensor,
  # use tfds.features.Sequence(per_point_feature). Also consider using
  # tfds.features.ClassLabel instead of int32 for semantic category.
  result = spec_dict.SpecDict({
      'positions':
          tfds.features.Tensor(shape=(None, 3), dtype=tf.float32),
      'point_identifiers':
          tfds.features.Tensor(shape=(None, 1), dtype=tf.int64),
      'timestamps':
          tfds.features.Tensor(shape=(None, 1), dtype=tf.float32),
  })
  # TODO(epot): Replace by ClassLabel
  result.maybe_set(
      'category_labels',
      category_labels,
      tfds.features.Tensor(shape=(None, 1), dtype=tf.int32),
  )
  return result
