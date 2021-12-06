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

"""Nerf task."""

import dataclasses
import enum
from typing import Any, Callable, Optional, Tuple, Union

from sunds import core
from sunds import utils
from sunds.core import specs
from sunds.core import tf_geometry
from sunds.tasks import boundaries_utils
from sunds.typing import FeatureSpecsHint, Split, TensorDict, TreeDict  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


class YieldMode(enum.Enum):
  """Define the example structure yield by `tf.data.Dataset`.

  Attributes:
    RAY: Each example is an individual ray. Cameras and image dims flattened.
    IMAGE: Each example is an indidual camera `(h, w)` (default).
    STACKED: Each example contain all camera stacked together
      `(num_cams, h, w)`.
    DICT: The example contain the dict of all cameras (
      `'cameras': {'camera0': {'ray_origins': ...}, 'camera1': ...}`).
  """

  # TODO(epot): Migrate to coutils.epy.EnumStr
  def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
    return name.lower()

  RAY = enum.auto()
  IMAGE = enum.auto()
  STACKED = enum.auto()
  DICT = enum.auto()


def _camera_specs(
    original_specs: FeatureSpecsHint,
    additional_camera_specs: FeatureSpecsHint,
) -> FeatureSpecsHint:
  """Fetch the camera."""
  # Minimal camera position/rotation
  spec = specs.camera_spec()
  # Always load rgb
  spec['color_image'] = True

  # If ray_direction/origin are present, use them.
  for additional_keys in [
      'ray_directions',
      'ray_origins',
  ]:
    spec[additional_keys] = additional_keys in original_specs

  # Fetch additional keys on demand
  spec.update(additional_camera_specs)
  return spec


@dataclasses.dataclass(frozen=True)
class CenterNormalizeParams:
  """Normalization params for centering.

  If passed to `Nerf(normalize_rays=)`, each rays origin will be shifted
  by a center value.
  Centering will be performed based on the axis aligned bounding box of the
  target camera frustrum and the origins of the other camera.

  Attributes:
    far_plane: far plane to use when calculating frustrums.
    jitter: Translational jitter to apply after centering
  """
  far_plane: float
  jitter: float = 3.0


@dataclasses.dataclass(frozen=True)
class Nerf(core.FrameTask):
  """Nerf-like processing.

  Compute the non-normalized ray origins and directions and returns the dict:

  ```python
  {
      'ray_origins': tf.Tensor(shape=(*batch_shape, 3), dtype=tf.float32),
      'ray_directions': tf.Tensor(shape=(*batch_shape, 3), dtype=tf.float32),
      'color_image': tf.Tensor(shape=(*batch_shape, 3), dtype=tf.uint8),
      'scene_name': tf.Tensor(shape=(1,), dtype=tf.string),
      'frame_name': tf.Tensor(shape=(1,), dtype=tf.string),
      'camera_name': tf.Tensor(shape=(1,), dtype=tf.string),
  }
  ```

  `batch_shape` is:

  * `(num_cameras, h, w)` if yield_mode is STACKED
  * `(h, w)` if yield_mode is IMAGE (default)
  * `()` if yield_mode is RAY

  Note:

  * If the dataset has multiple cameras, each camera will be yielded
    individually, unless `yield_mode` is set to another value.
  * `ray_origins`/`ray_directions` are automatically computed if not present.
  * `ray_directions` may contain invalid values (all-zeros) for some datasets.

  Attributes:
    yield_mode: Control whether the dataset returns examples as individual
      `RAY`, individual `IMAGE` (default), all camera stacked (STACKED)
      together. Accept `str` or `sunds.tasks.YieldMode`.
    normalize_rays: If True, the scene is scaled such as all objects in the
      scenes are contained in a `[-1, 1]` box. Ray directions are also
      normalized. Can also be a configurable dataclass for more options.
    remove_invalid_rays: If True (default), do not yield rays where
      ray_direction == [0,0,0]. Only has effect when `yield_mode=RAY`.
    additional_camera_specs: Additional camera specs to include. Those features
      can be transformed by other option (e.g. `yield_mode=RAY` will
      unbatch `category_image`,...).
    additional_frame_specs: Additional features specs to include. Those features
      will be forwarded as-is (without any transformation). Should not contain
      any camera fields.
  """
  yield_mode: Union[str, YieldMode] = YieldMode.IMAGE
  normalize_rays: Union[bool, CenterNormalizeParams] = False
  remove_invalid_rays: bool = True
  additional_camera_specs: FeatureSpecsHint = dataclasses.field(
      default_factory=dict)
  additional_frame_specs: FeatureSpecsHint = dataclasses.field(
      default_factory=dict)

  def __post_init__(self):
    # Normalize arguments
    object.__setattr__(self, 'yield_mode', YieldMode(self.yield_mode))
    object.__setattr__(
        self,
        'additional_camera_specs',
        _normalize_additional_specs(self.additional_camera_specs),
    )
    object.__setattr__(
        self,
        'additional_frame_specs',
        _normalize_additional_specs(self.additional_frame_specs),
    )
    assert 'cameras' not in self.additional_frame_specs

  def as_dataset(self, **kwargs):
    # Forward the split name to the pipeline function
    assert kwargs.get('batch_size') is None, 'Batch size incompatible with Nerf'
    return super().as_dataset(
        pipeline_kwargs=dict(split=kwargs['split']), **kwargs)

  @property
  def frame_specs(self) -> Optional[FeatureSpecsHint]:
    """Expected specs of the scene understanding pipeline."""
    if 'cameras' in self.additional_frame_specs:
      raise ValueError(
          '`additional_frame_specs` should not contain cameras specs. '
          'Please use `additional_camera_specs` instead. Got: '
          f'{self.additional_frame_specs}')

    curr_specs = {
        # Extract the scene name for scene boundaries
        'scene_name': tfds.features.Text(),
        'frame_name': tfds.features.Text(),
        'pose': specs.pose_spec(),
        'cameras': {  # pylint: disable=g-complex-comprehension
            camera_name: _camera_specs(
                camera_spec,
                additional_camera_specs=self.additional_camera_specs,
            ) for camera_name, camera_spec in
            self.full_frame_specs['cameras'].items()
        },
    }
    curr_specs.update(self.additional_frame_specs)
    return curr_specs

  def pipeline(self, ds: tf.data.Dataset, *, split: Split) -> tf.data.Dataset:
    """Post processing specs."""
    # Apply the transformations:
    # * Eventually compute the rays (if not included)
    # * Eventually normalize the rays
    # * Unbatch to yield individual images or rays (if the dataset has multiple
    #   cameras)

    # Eventually include rays (if not included)
    ds = ds.map(
        _add_rays(additional_camera_specs=self.additional_camera_specs),  # pylint: disable=no-value-for-parameter
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Eventually normalize the rays
    if self.normalize_rays:
      ds = _normalize_rays_ds(
          ds,
          split=split,
          normalize_rays=self.normalize_rays,
          scene_builder=self.scene_builder,
      )

    # Eventually flatten the camera, images
    if self.yield_mode in {YieldMode.RAY, YieldMode.IMAGE}:
      ds = _flatten_camera_dim_ds(ds)
    elif self.yield_mode == YieldMode.STACKED:
      ds = ds.map(_stack_cameras)

    # Eventually flatten the images. Batch size: `(h, w)` -> `()`
    if self.yield_mode == YieldMode.RAY:
      if self.additional_frame_specs:
        raise NotImplementedError(
            'Additional frame specs not compatible with YieldMode.RAY. '
            'Please open a GitHub issue if needed.')
      # Copy static fields (scene_name, frame_name, ...) to each ray.
      ds = ds.map(_clone_static_fields)

      # Remove (H, W, ...) part of shape.
      ds = ds.unbatch().unbatch()

      # Drop invalid rays (direction == 0).
      if self.remove_invalid_rays:
        ds = ds.filter(_has_valid_ray_directions)

    return ds


@utils.map_fn
def _add_rays(
    frame: TensorDict,
    *,
    additional_camera_specs: FeatureSpecsHint,
) -> TensorDict:
  """Add the rays on all camera images."""
  # Get frame to scene transform.
  scene_from_frame = tf_geometry.Isometry(**frame.pop('pose'))

  cameras = {}
  # Compute the rays for each camera
  for camera_name, camera_data in frame['cameras'].items():
    ex = _add_rays_single_cam(camera_data, scene_from_frame=scene_from_frame)
    if 'intrinsics' not in additional_camera_specs:
      del ex['intrinsics']  # Exclude fields from the returned examples
    if 'extrinsics' not in additional_camera_specs:
      del ex['extrinsics']
    cameras[camera_name] = ex

  # Update frame
  frame['cameras'] = cameras
  return frame


def _add_rays_single_cam(
    camera_data: TensorDict,
    *,
    scene_from_frame: tf_geometry.Isometry,
) -> TensorDict:
  """Returns the camera, eventually with the rays added."""
  if _has_precomputed_rays(camera_data):
    return camera_data
  else:
    # Logic below for generating camera rays only applies to perspective
    # cameras. It will produce incorrect camera rays for other types of
    # cameras (e.g. those with distortions).
    camera_type = camera_data['intrinsics']['type']
    tf.debugging.assert_equal(camera_type, 'PERSPECTIVE')

    # Pinhole camera model below does not know how to handle lens distortion.
    # Ensure that no distortion exists here.
    radial_distortion = camera_data['intrinsics']['distortion']['radial']
    tf.debugging.assert_near(radial_distortion,
                             tf.zeros_like(radial_distortion))
    tangential_distortion = (
        camera_data['intrinsics']['distortion']['tangential'])
    tf.debugging.assert_near(tangential_distortion,
                             tf.zeros_like(tangential_distortion))

    h, w, _ = camera_data['color_image'].shape

    # Compute camera pose w.r.t scene (camera to scene transform).
    camera_from_frame = tf_geometry.Isometry(**camera_data['extrinsics'])
    scene_from_camera = scene_from_frame * camera_from_frame.inverse()

    # Get rays w.r.t scene passing through every pixel center of the camera.
    camera_intrinsics = camera_data['intrinsics']
    ray_origins, ray_directions = tf_geometry.rays_from_image_grid(
        camera=tf_geometry.PinholeCamera(
            K=camera_intrinsics['K'],
            # Use static shape if available
            image_width=w or camera_intrinsics['image_width'],
            image_height=h or camera_intrinsics['image_height']),
        world_from_camera=scene_from_camera,
    )

    camera_data['ray_origins'] = ray_origins
    camera_data['ray_directions'] = ray_directions
    return camera_data


def _has_precomputed_rays(camera_dict: TreeDict[Any]) -> bool:
  """Returns True."""
  return all(k in camera_dict for k in ('ray_origins', 'ray_directions'))


def _normalize_rays_ds(
    ds: tf.data.Dataset,
    *,
    split: Split,
    normalize_rays: Union[bool, CenterNormalizeParams],
    scene_builder: tfds.core.DatasetBuilder,
) -> tf.data.Dataset:
  """Normalize the rays."""
  if isinstance(normalize_rays, bool):
    assert normalize_rays
    scene_boundaries = _get_scene_boundaries(scene_builder, split=split)
    normalize_fn = _normalize_rays(scene_boundaries=scene_boundaries)  # pylint: disable=no-value-for-parameter
    normalize_fn = _apply_to_all_cameras(fn=normalize_fn)  # pylint: disable=no-value-for-parameter
    ds = ds.map(
        normalize_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
  elif isinstance(normalize_rays, CenterNormalizeParams):
    # TODO(tutmann): Consider moving this down below keep_as_image.
    # We then would normalize only the rays that we actually query.
    # This would have an additional free data augmentation effect on things.
    ds = ds.map(
        _center_example(  # pylint: disable=no-value-for-parameter
            far_plane_for_centering=normalize_rays.far_plane,
            jitter=normalize_rays.jitter,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
  else:
    raise TypeError(f'Invalid `normalize_rays`: {normalize_rays!r}')
  return ds


@utils.map_fn
def _normalize_rays(
    static_ex: TensorDict,
    camera_ex: TensorDict,
    *,
    scene_boundaries: boundaries_utils.MultiSceneBoundaries,
) -> TensorDict:
  """Normalize the rays origins and direction."""
  origins = camera_ex['ray_origins']
  directions = camera_ex['ray_directions']

  min_corner, max_corner = scene_boundaries.get_corners(static_ex['scene_name'])

  # Rescale (x, y, z) from [min, max] -> [-1, 1]
  origins = utils.interp(
      origins,
      from_=(min_corner, max_corner),
      to=(-1., 1.),
      axis=-1,
  )
  # We also need to rescale the camera direction by size.
  # The direction can be though of a ray from a point in space (the camera
  # origin) to another point in space (say the red light on the lego
  # bulldozer). When we scale the scene in a certain way, this direction
  # also needs to be scaled in the same way.
  # Scale each (x, y, z) by (max - min) / 2
  directions = directions * 2 / (max_corner - min_corner)

  # Normalize the rays. WARNING: direction == 0 for invalid rays.
  directions = tf.math.divide_no_nan(
      directions, tf.norm(directions, axis=-1, keepdims=True))

  # TODO(epot): If present, should also scale depth accordingly, so
  # `depth * direction` still works
  if 'depth_image' in camera_ex:
    raise NotImplementedError(
        'Depth not normalized for now. Please open a Github issue if you '
        'need this feature.')

  camera_ex['ray_origins'] = origins
  camera_ex['ray_directions'] = directions
  return camera_ex


def _get_scene_boundaries(
    scene_builder: tfds.core.DatasetBuilder,
    split: str,
) -> boundaries_utils.MultiSceneBoundaries:
  """Extract the scenes boundaries."""
  # Each scene examples can be huge, so disable parallelism to avoid
  # OOM errors (especially on Colab)
  read_config = tfds.ReadConfig(
      interleave_cycle_length=1,
      interleave_block_length=None,
  )
  ds = scene_builder.as_dataset(
      split=split,
      # Only decode the scene boundaries
      decoders=tfds.decode.PartialDecoding({
          'scene_name': tfds.features.Text(),
          'scene_box': specs.aligned_box_3d_spec(),
      }),
      read_config=read_config,
  )
  # Load all scenes boundaries
  # TODO(epot): This can be very costly. Should have a way to indicate
  # all datasets share the same scene boundaries, so only the first one
  # is required.
  ds = tfds.as_numpy(ds)
  ds = utils.tqdm(ds, desc='Loading scenes...', leave=False)
  ds = list(ds)
  scene_boundaries = boundaries_utils.MultiSceneBoundaries(ds)
  return scene_boundaries


@utils.map_fn
def _apply_to_all_cameras(
    ex: TensorDict,
    *,
    fn: Callable[[TensorDict, TensorDict], TensorDict],
) -> TensorDict:
  """Apply the given function individually on each individual camera.

  Args:
    ex: Original example
    fn: Function with signature (static_ex, camera_ex) -> camera_ex. The `ex` is
      decomposed into static_ex (values shared across all camera, like
      `scene_name`) and `camera_ex` (camera specific values)

  Returns:
    The `ex` with the `fn` transformation applied to all cameras.
  """
  ex['cameras'] = {
      name: fn(ex, camera) for name, camera in ex['cameras'].items()
  }
  return ex


def _stack_cameras(frame: TensorDict) -> TensorDict:
  """Batch camera together."""
  frame, cameras = _extract_static_and_camera_dict(frame)
  assert not set(frame).intersection(cameras)
  return {**frame, **cameras}


def _flatten_camera_dim_ds(ds: tf.data.Dataset,) -> tf.data.Dataset:
  """Yield camera individually."""
  num_elem = len(ds)
  num_cameras = len(ds.element_spec['cameras'])
  ds = ds.interleave(
      _flatten_camera_dim,
      cycle_length=16,  # Hardcoded values for determinism
      block_length=16,
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  # Update the number of element in the dataset (ds.interleave loose the info)
  num_elem = num_elem * num_cameras
  ds = ds.apply(tf.data.experimental.assert_cardinality(num_elem))
  return ds


@utils.map_fn
def _flatten_camera_dim(frame: TensorDict) -> tf.data.Dataset:
  """Add the rays on all camera images."""
  frame, cameras = _extract_static_and_camera_dict(frame)
  return _ds_merge_dict(
      slices=cameras,
      static=frame,
  )


def _extract_static_and_camera_dict(
    frame: TensorDict) -> Tuple[TensorDict, TensorDict]:
  """Extract the static features, and stack the camera features."""
  frame = dict(frame)
  cameras = frame.pop('cameras')
  camera_names, cameras = zip(*cameras.items())

  # Transpose list[dict] into dict[list]
  cameras = tf.nest.map_structure(lambda *args: list(args), *cameras)
  cameras['camera_name'] = tf.convert_to_tensor(camera_names)
  return frame, cameras


def _ds_merge_dict(
    slices: TensorDict,
    static: TensorDict,
) -> tf.data.Dataset:
  """Returns a new dataset containing elements.

  Args:
    slices: Dict of the dynamic elements of the dataset
    static: Dict of the static element of the dataset (repeated across examples)

  Returns:
    The new dataset dict (slices and static dict merged).
  """
  slices_ds = tf.data.Dataset.from_tensor_slices(slices)

  if not static:
    return slices_ds
  static_ds = tf.data.Dataset.from_tensors(static)
  static_ds = static_ds.repeat()
  ds = tf.data.Dataset.zip((slices_ds, static_ds))
  ds = ds.map(lambda x, y: {**x, **y})
  return ds


@utils.map_fn
def _clone_static_fields(ex: TensorDict,) -> TensorDict:
  """Clone static fields to each ray.

  Args:
    ex: A single-camera or multi-camera example. Must have the following fields
      -- frame_name, scene_name.

  Returns:
    Modified version of `ex` with `*_name` features cloned once per pixel.
  """
  # Identify batch shape.
  batch_shape: tf.TensorShape = ex['color_image'].shape[0:-1]

  # TODO(duckworthd): Duplicate ALL static fields, including those specified
  # in additional_frame_specs.
  def _clone(v):
    return tf.fill(batch_shape, v)

  # Clone individual fields.
  ex['scene_name'] = _clone(ex['scene_name'])
  ex['frame_name'] = _clone(ex['frame_name'])
  ex['camera_name'] = _clone(ex['camera_name'])
  return ex


def _normalize_additional_specs(spec: FeatureSpecsHint) -> FeatureSpecsHint:
  """Normalize feature specs."""
  if isinstance(spec, (list, tuple, set)):
    return {k: True for k in spec}
  if not isinstance(spec, dict):
    raise TypeError(f'Invalid additional spec type: {type(spec)}')
  return spec


def _has_valid_ray_directions(ex: TensorDict) -> TensorDict:
  directions = ex['ray_directions']
  return tf.logical_not(tf.reduce_all(directions == 0.))


@utils.map_fn
def _center_example(
    ex: TensorDict,
    *,
    far_plane_for_centering: float,
    jitter: float,
) -> TensorDict:
  """Centers the example.

  Centering will be performed based on the axis aligned bounding box of
  the target view frustrum and the origins of the input views.

  Args:
    ex: Original example
    far_plane_for_centering: far plane to use when calculating frustrums.
    jitter: translational jitter to apply after centering.

  Returns:
    The `ex` with all camera origins centered.
  """
  points = []
  for camera in ex['cameras'].values():
    origins = tf.reshape(camera['ray_origins'], (-1, 3))
    directions = tf.reshape(camera['ray_directions'], (-1, 3))

    # Filter invalid rays.
    valid_directions = tf.reduce_any(tf.math.not_equal(directions, 0.), axis=-1)
    origins = tf.boolean_mask(origins, valid_directions)
    directions = tf.boolean_mask(directions, valid_directions)

    points.append(origins)
    points.append(origins + directions * far_plane_for_centering)
  points = tf.concat(points, axis=0)
  assert len(points.shape) == 2
  points_min = tf.reduce_min(points, axis=0)
  points_max = tf.reduce_max(points, axis=0)
  bbox_size = points_max - points_min
  center = points_min + bbox_size / 2
  center += tf.random.truncated_normal(shape=center.shape, stddev=jitter)

  for camera in ex['cameras'].values():
    camera['ray_origins'] -= center
  return ex
