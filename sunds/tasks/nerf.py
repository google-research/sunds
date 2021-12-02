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
from typing import Any, Callable, Optional, Union

from sunds import core
from sunds import utils
from sunds.core import specs
from sunds.core import tf_geometry
from sunds.tasks import boundaries_utils
from sunds.typing import FeatureSpecs, Split, TensorDict, TreeDict  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


def _camera_specs(
    original_specs: FeatureSpecs,
    additional_camera_specs: FeatureSpecs,
) -> FeatureSpecs:
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
      'scene_name': tf.Tensor(shape=(*batch_shape, 1), dtype=tf.string),
      'frame_name': tf.Tensor(shape=(*batch_shape, 1), dtype=tf.string),
      'camera_name': tf.Tensor(shape=(*batch_shape, 1), dtype=tf.string),
      ...,
  }
  ```

  `batch_shape` is:

  * `(h, w)` if `keep_as_image=True` (default)
  * `()` if `keep_as_image=False`

  Note:

  * If the dataset has multiple cameras, each camera will be yielded
    individually, unless `yield_individual_camera=False`.
  * `ray_origins`/`ray_directions` are automatically computed if not present.
  * `camera_name` feature is only available when `yield_individual_camera=True`
    (default).
  * `ray_directions` may contain invalid values (all-zeros) when
    `keep_as_image=True`.

  Attributes:
    keep_as_image: Whether to keep the image dimension.
    normalize_rays: If True, the scene is scaled such as all objects in the
      scenes are contained in a `[-1, 1]` box. Ray directions are also
      normalized. Can also be a configurable dataclass for more options.
    yield_individual_camera: If True (default), each camera within a frame is
      yield individually. Otherwise, each example contains all cameras (
      `'cameras': {'camera0': {'ray_origins': ...}, 'camera1': ...}`).
    remove_invalid_rays: If True (default), do not yield rays where
      ray_direction == [0,0,0]. Only has effect when keep_as_image=False and
      yield_individual_camera=True.
    additional_camera_specs: Additional camera specs to include. Those features
      can be transformed by other option (e.g. `keep_as_image=False` will
      unbatch `category_image`,...).
    additional_frame_specs: Additional features specs to include. Those features
      will be forwarded as-is (without any transformation). Should not contain
      any camera fields.
  """
  keep_as_image: bool = True
  normalize_rays: Union[bool, CenterNormalizeParams] = False
  yield_individual_camera: bool = True
  remove_invalid_rays: bool = True
  additional_camera_specs: FeatureSpecs = dataclasses.field(
      default_factory=dict)
  additional_frame_specs: FeatureSpecs = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    # Normalize additional_specs
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

  def as_dataset(self, **kwargs):
    # Forward the split name to the pipeline function
    return super().as_dataset(
        pipeline_kwargs=dict(split=kwargs['split']), **kwargs)

  @property
  def frame_specs(self) -> Optional[FeatureSpecs]:
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
    # Apply the pre-processing:
    # * Eventually compute the rays (if not included)
    # * Yield individual images (if the dataset has multiple camera)
    num_elem = len(ds)
    ds = ds.interleave(
        _process_frame(yield_individual_camera=self.yield_individual_camera),  # pylint: disable=no-value-for-parameter
        cycle_length=16,  # Hardcoded values for determinism
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Update the number of element in the dataset (ds.interleave loose the info)
    if self.yield_individual_camera:
      num_cameras = len(self.frame_specs['cameras'])
      num_elem = num_elem * num_cameras
    ds = ds.apply(tf.data.experimental.assert_cardinality(num_elem))

    # Eventually normalize the rays
    if isinstance(self.normalize_rays, bool):
      if self.normalize_rays:
        scene_boundaries = _get_scene_boundaries(
            self.scene_builder, split=split)
        normalize_fn = _normalize_rays(scene_boundaries=scene_boundaries)  # pylint: disable=no-value-for-parameter
        normalize_fn = _apply_to_all_cameras(  # pylint: disable=no-value-for-parameter
            fn=normalize_fn,
            yield_individual_camera=self.yield_individual_camera,
        )
        ds = ds.map(
            normalize_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    elif isinstance(self.normalize_rays, CenterNormalizeParams):
      if self.yield_individual_camera:
        raise ValueError('centering require yield_individual_camera=False')
      # TODO(tutmann): Consider moving this down below keep_as_image.
      # We then would normalize only the rays that we actually query.
      # This would have an additional free data augmentation effect on things.
      ds = ds.map(
          _center_example(  # pylint: disable=no-value-for-parameter
              far_plane_for_centering=self.normalize_rays.far_plane,
              jitter=self.normalize_rays.jitter,
          ),
          num_parallel_calls=tf.data.AUTOTUNE,
      )
    else:
      raise TypeError(f'Invalid `normalize_rays`: {self.normalize_rays!r}')

    # Eventually flatten the images. Batch size: `(h, w)` -> `()`
    if not self.keep_as_image:
      if self.additional_frame_specs:
        raise ValueError(
            'Additional frame specs not compatible with keep_as_image=False')
      # Copy static fields (scene_name, frame_name, ...) to each ray.
      ds = ds.map(
          _clone_static_fields(  # pylint: disable=no-value-for-parameter
              yield_individual_camera=self.yield_individual_camera,))

      # Remove (H, W, ...) part of shape.
      ds = ds.unbatch().unbatch()

      # Drop invalid rays (direction == 0).
      if self.remove_invalid_rays and self.yield_individual_camera:
        ds = ds.filter(_has_valid_ray_directions)

    return ds


@utils.map_fn
def _process_frame(
    frame: TensorDict,
    *,
    yield_individual_camera: bool,
) -> tf.data.Dataset:
  """Add the rays on all camera images."""
  # Get frame to scene transform.
  scene_from_frame = tf_geometry.Isometry(**frame.pop('pose'))

  cameras = {}
  # Compute the rays for each camera
  for camera_name, camera_data in frame['cameras'].items():
    ex = _add_rays_single_cam(camera_data, scene_from_frame=scene_from_frame)
    del ex['intrinsics']  # Exclude fields from the returned examples
    del ex['extrinsics']
    cameras[camera_name] = ex

  if yield_individual_camera:
    camera_names, cameras = zip(*cameras.items())

    # Transpose list[dict] into dict[list]
    cameras = tf.nest.map_structure(lambda *args: list(args), *cameras)
    cameras['camera_name'] = tf.convert_to_tensor(camera_names)

    frame.pop('cameras')
    return _ds_merge_dict(
        slices=cameras,
        static=frame,
    )
  else:
    # Otherwise, keep the camera names / structure
    frame['cameras'] = cameras
    return tf.data.Dataset.from_tensors(frame)


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
    raise NotImplementedError('Depth not normalized for now.')

  camera_ex['ray_origins'] = origins
  camera_ex['ray_directions'] = directions
  return camera_ex


def _get_scene_boundaries(
    scene_builder,
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
    yield_individual_camera: bool,
) -> TensorDict:
  """Apply the given function individually on each individual camera.

  Args:
    ex: Original example
    fn: Function with signature (static_ex, camera_ex) -> camera_ex. The `ex` is
      decomposed into static_ex (values shared across all camera, like
      `scene_name`) and `camera_ex` (camera specific values)
    yield_individual_camera: Whether `ex` contain all cameras or a single one.

  Returns:
    The `ex` with the `fn` transformation applied to all cameras.
  """
  if not yield_individual_camera:
    ex['cameras'] = {
        name: fn(ex, camera) for name, camera in ex['cameras'].items()
    }
    return ex
  else:
    return fn(ex, ex)


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
def _clone_static_fields(
    ex: TensorDict,
    *,
    yield_individual_camera: bool,
) -> TensorDict:
  """Clone static fields to each ray.

  Args:
    ex: A single-camera or multi-camera example. Must have the following fields
      -- frame_name, scene_name.
    yield_individual_camera: If True, field camera_name is also required.

  Returns:
    Modified version of `ex` with `*_name` features cloned once per pixel.
  """
  # Identify batch shape.
  if yield_individual_camera:
    camera_ex = ex
  else:
    camera_ex = next(iter(ex['cameras'].values()))
  batch_shape: tf.TensorShape = camera_ex['color_image'].shape[0:-1]

  # TODO(duckworthd): Duplicate ALL static fields, including those specified
  # in additional_frame_specs.
  def _clone(v):
    return tf.fill(batch_shape, v)

  # Clone individual fields.
  ex['scene_name'] = _clone(ex['scene_name'])
  ex['frame_name'] = _clone(ex['frame_name'])
  if yield_individual_camera:
    ex['camera_name'] = _clone(ex['camera_name'])

  return ex


def _normalize_additional_specs(spec: FeatureSpecs) -> FeatureSpecs:
  """Normalize feature specs."""
  if isinstance(spec, (list, set)):
    return {k: True for k in spec}
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
