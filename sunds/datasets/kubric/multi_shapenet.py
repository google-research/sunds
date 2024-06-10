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

r"""MultiShapeNet dataset generator.

This file provides SceneRenderer, a class for generating random scenes
containing ShapeNet objects.

# Usage

```
config = multi_shapenet.SceneConfig()
renderer = multi_shapenet.SceneRenderer(config)
result = renderer.render_scene(seed=42, split="train")
```

"""

# pylint: disable=logging-format-interpolation

import dataclasses
import enum
import functools
import logging
import os
import pathlib
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from etils import epy
import kubric as kb
import kubric.randomness as kbrand
from kubric.renderer import blender
import kubric.simulator.pybullet
import numpy as np
import pandas as pd
import pyquaternion
from sunds.datasets.kubric import base


class ObjectSamplingStrategy(enum.Enum):
  UNIFORM = "uniform"
  STRATIFIED_BY_CATEGORY = "stratified_by_category"

  def __str__(self):
    return str(self.value)


class ObjectPlacementSamplingStrategy(enum.Enum):
  UNIFORM = "uniform"
  UPRIGHT = "upright"

  def __str__(self):
    return str(self.value)


class BackgroundSamplingStrategy(enum.Enum):
  HDRI = "hdri"
  WHITE = "white"

  def __str__(self):
    return str(self.value)


class CameraSamplingStrategy(epy.StrEnum):
  SPHERE_SHELL = enum.auto()
  SPHERE_SHELL_MIXED_DISTANCE = enum.auto()
  CIRCLE = enum.auto()


@dataclasses.dataclass
class SceneConfig(base.SceneConfig):
  """Configuration for Kubric Scenes."""

  # Number of objects in the scene.
  min_num_objects: int = 16
  max_num_objects: int = 32

  # Source for foreground objects and backgrounds, respectively.
  assets_manifest: str = (
      "gs://kubric-public/assets/ShapeNetCore.v2.json")
  hdri_manifest: str = "gs://kubric-public/assets/HDRI_haven.json"
  kubasic_manifest: str = "gs://kubric-public/assets/KuBasic.json"

  # Height and width of rendered images in pixels
  resolution: Tuple[int, int] = (256, 256)

  # Controls the number of frames to generate per scene.
  frame_start: int = 1
  frame_end: int = 8
  frame_rate: int = 12

  # number of simulation steps per second
  step_rate: int = 240

  # Perspective camera parameters.
  focal_length: float = 35
  sensor_width: float = 32

  min_camera_height: float = 0.0

  # Region in which objects will spawn. Note that a large "floor" is always
  # created at z=0. The scene boundaries are controlled by this parameter.
  spawn_region: List[Tuple[int, int, int]] = dataclasses.field(
      default_factory=lambda: [(-5, -5, 2), (5, 5, 64)]
  )

  # Sampling strategy used when sampling assets. Must be one of the following
  # options,
  #   UNIFORM: sample assets uniformly at random.
  #   STRATIFIED_BY_CATEGORY: sample a category uniformly at random, then
  #     sample an asset from that category.
  object_sampling_strategy: ObjectSamplingStrategy = (
      ObjectSamplingStrategy.UNIFORM
  )

  # Sampling strategy used when sampling backgrounds. Must be one of the
  # following options,
  #   HDRI: sample a random HDRI dome.
  #   WHITE: all scenes have a solid-white background.
  background_sampling_strategy: BackgroundSamplingStrategy = (
      BackgroundSamplingStrategy.HDRI
  )

  # Strategy used to sample the camera poses. Must be one of the following:
  #  SPHERE_SHELL: randomly sample inside a half-sphere shell
  #  CIRCLE: equally spaced along a constant (but random) height circle.
  camera_sampling_strategy: CameraSamplingStrategy = (
      CameraSamplingStrategy.SPHERE_SHELL
  )

  # Will split backgrounds in test/train if enabled.
  split_backgrounds: bool = False

  # If specified, the list of valid object categories to choose assets from.
  object_category_allowlist: Optional[List[str]] = None

  # Maximum number of assets for each semantic category. If unspecified, no
  # limit is placed.
  max_num_objects_per_category: Optional[int] = None

  # Strategy used to sample objects' spawned position and orientation. Must
  # be one of the following options,
  #   UNIFORM: anywhere in the spawn region.
  #   UPRIGHT: directly above the ground with the z-axis fixed.
  object_placement_sampling_strategy: ObjectPlacementSamplingStrategy = (
      ObjectPlacementSamplingStrategy.UNIFORM
  )

  # Number of steps to run simulation before rendering
  num_steps_simulation: int = 100

  fix_obj_seed: Optional[int] = None
  fix_bg_seed: Optional[int] = None
  fix_cam_seed: Optional[int] = None


def _get_scene_boundaries(
    floor: kb.Cube,
    spawn_region: List[Tuple[float, float, float]],
    foreground_objects: List[kb.Object3D],
) -> Dict[str, List[int]]:
  """Constructs bounding box for all scene-varying content.

  This bounding box ostensibly includes the objects themselves and their
  shadows. Due to complexity, a guarantee on this cannot be made. The logic
  here is best-effort.

  Args:
    floor: Cube representing the scene's floor. The cube is axis-aligned with
      its top surface lying below all objects in the scene.
    spawn_region: Lower and upper boundaries of a 3D axis-aligned box, inside
      which all foreground objects are guaranteed to lie within.
    foreground_objects: Kubric objects in the scene.

  Returns:
    Dict with two entries, "min" and "max". These entries are 3-element lists
      containing the lower and upper boundaries of a 3D axis-aligned bounding
      box.
  """
  scene_min, scene_max = np.asarray(spawn_region)

  # Bottom of scene is top of floor.
  floor_max = np.asarray(floor.position) + np.asarray(floor.scale) / 2
  scene_min[2] = floor_max[2]

  # Top of scene is half as high as the scene is long/wide.
  dx = scene_max[0] - scene_min[0]
  dy = scene_max[1] - scene_min[1]
  dz = max(dx, dy) / 2.0
  scene_max[2] = scene_min[2] + dz

  # Define some amount of buffer volume to the scene's bounding box to
  # ensure objects and shadows lie in the scene's bounding box.
  boundary_epsilon = float(np.sqrt(3))  # diagonal length of a unit cube
  scene_min = scene_min - boundary_epsilon
  scene_max = scene_max + boundary_epsilon

  # Ensure that all objects lie within the [scene_min, scene_max]. Note:
  # object AABB boxes are *loose*. They are calculated based on the object's
  # non-axis aligned bounding box. Don't be surprised if you see an object's
  # AABB box has a z-value < 0.
  #
  # shape=(num_objects, 2, 3)
  assert foreground_objects, "Must have at least one foreground object."
  object_aabboxes = np.array([obj.aabbox for obj in foreground_objects])

  # shape=(3,)
  scene_aabbox_min = np.min(object_aabboxes[:, 0, :], axis=0)
  scene_aabbox_max = np.max(object_aabboxes[:, 1, :], axis=0)
  if not (
      np.all(scene_min <= scene_aabbox_min)
      and np.all(scene_max >= scene_aabbox_max)
  ):
    raise base.IrrecoverableRenderingError(
        "Foreground objects do not lie in the desired scene bounding box.\n"
        f"  Desired scene bounds: [{scene_min}, {scene_max}]\n"
        f"  Actual scene bounds: [{scene_aabbox_min}, {scene_aabbox_max}]"
    )

  return {
      "min": list(scene_min),
      "max": list(scene_max),
  }


def _sample_up_to(
    df: pd.DataFrame, *, n: int, random_state: np.random.RandomState
) -> pd.DataFrame:
  """Sample up to 'n' random entries from 'df'."""
  if n >= len(df):
    return df
  return df.sample(n, replace=False, random_state=random_state)


def _reset_position_and_orientation(
    obj: kb.Object3D,
    rng: np.random.RandomState,
    *,
    position: np.ndarray,
    quaternion: np.ndarray,
):
  """Resets the position and quaternion of an object to its defaults."""
  del rng
  obj.position = position
  obj.quaternion = quaternion


def _random_rotation(
    obj: kb.Object3D, rng: np.random.RandomState, *, axis: Optional[str]
):
  """Applies a random rotation to an object."""
  # Unlike kbrandom.rotation_sampler, this code COMPOSES the random rotation
  # with the existing object rotation.
  original = pyquaternion.Quaternion(obj.quaternion)
  rotation = pyquaternion.Quaternion(kbrand.random_rotation(axis=axis, rng=rng))
  obj.quaternion = tuple((rotation * original).normalised)


def _random_xy_sampler(
    obj: kb.Object3D,
    rng: np.random.RandomState,
    *,
    spawn_region: List[Tuple[int, int, int]],
):
  """Sample an object's position such that it's sitting on the floor.

  Args:
    obj: Object to set position for.
    rng: Random number generator.
    spawn_region: AABBox defining region in which the object should exist.
  """
  # Identify the set of valid positions inside of 'spawn_region' such that
  # the object's AABBox would not intersect the spawn region's limits.
  rotated_bounds = obj.aabbox
  effective_spawn_region = np.array(spawn_region) - rotated_bounds

  # Ensure that the spawn region is bigger than the object.
  assert np.all(effective_spawn_region[0] <= effective_spawn_region[1])

  # Set object position. Z-value is set to be the smallest allowable value.
  x, y, _ = rng.uniform(*effective_spawn_region)
  z = effective_spawn_region[0, 2] + 1e-5
  obj.position = np.array([x, y, z]) + obj.position


def _is_in_spawn_region(
    obj: kb.Object3D,
    *,
    spawn_region: List[Tuple[float, float, float]],
) -> bool:
  """Checks if object's AABBox is within the spawn region."""
  scene_lower, scene_upper = np.array(spawn_region)
  obj_lower, obj_upper = obj.aabbox
  result = np.all((scene_lower <= obj_lower) & (scene_upper >= obj_upper))
  return result


def _place_upright_object(
    asset, simulator, spawn_region, rng, *, max_trials=1000
):
  """Randomly places an upright object on the floor."""
  # ShapeNet places y=up. This transformation ensures z=up by rotating about
  # the x-axis by 90 degrees before the random rotation is chosen.
  initial_quaternion = pyquaternion.Quaternion(
      axis=[1, 0, 0], radians=np.pi / 2
  )

  samplers = [
      functools.partial(
          _reset_position_and_orientation,
          position=asset.position,
          quaternion=list(initial_quaternion),
      ),
      functools.partial(_random_rotation, axis="Z"),
      functools.partial(_random_xy_sampler, spawn_region=spawn_region),
  ]

  def _is_invalid_position(obj: kb.Object3D) -> bool:
    is_in_spawn_region = _is_in_spawn_region(obj, spawn_region=spawn_region)
    is_intersecting_another_object = simulator.check_overlap(obj)
    is_valid_position = (
        is_in_spawn_region and not is_intersecting_another_object
    )
    if not is_in_spawn_region:
      logging.info(
          "Object %s is not in the spawn region. obj.aabbox=%s spawn_region=%s",
          obj.asset_id,
          obj.aabbox,
          spawn_region,
      )

    if is_intersecting_another_object:
      logging.info(
          "Object %s is intersecting another object in the scene.", obj.asset_id
      )

    return not is_valid_position

  return kbrand.resample_while(
      asset,
      samplers=samplers,
      condition=_is_invalid_position,
      max_trials=max_trials,
      rng=rng,
  )


def _normalize_object_scale(asset: kb.Object3D):
  """Normalize object scale such that it's AABBox's lengths are <=2."""
  min_bounds, max_bounds = asset.aabbox
  max_dim = np.max(max_bounds - min_bounds) + 1e-5
  asset.scale = asset.scale * 2 / max_dim


def _normalize_object_position(asset: kb.Object3D):
  """Normalize object position such that it is centered."""
  min_bounds, max_bounds = asset.aabbox
  center = (min_bounds + max_bounds) / 2
  asset.position = np.array(asset.position) - center


class SceneRenderer(base.BaseRenderer):
  """Renderer for a Kubric scene."""

  def __init__(self, config: SceneConfig):
    self._config = config
    self._scratch_dir = pathlib.Path(tempfile.mkdtemp())

    logging.info("Using scratch: %s", self._scratch_dir)
    self._setup_renderer()

    logging.info("Loading assets from %s", self._config.assets_manifest)
    self._asset_source = kb.AssetSource.from_manifest(
        self._config.assets_manifest
    )

    self._walls = []
    self._obj_rng = None
    self._cam_rng = None
    self._bg_rng = None

    ############################################################################
    # Choose which assets are eligible for use when building this dataset.

    # create a dataframe
    db = self._asset_source.db
    # convert category_id from str to int
    db["category_id"] = db["category_id"].astype(int)

    # Assign each asset to a test or train split.
    _, test_assets = self._asset_source.get_test_split(fraction=0.1)
    db["split"] = np.where(db["id"].isin(test_assets), "test", "train")

    # Limit assets to those in allowed categories.
    if config.object_category_allowlist:
      db = db[db["category"].isin(set(config.object_category_allowlist))]

    # Limit number of assets per category.
    if config.max_num_objects_per_category:
      db = (
          db.groupby("category_id")
          .apply(
              functools.partial(
                  _sample_up_to,
                  n=config.max_num_objects_per_category,
                  random_state=self._obj_rng,
              )
          )
          .reset_index(drop=True)
      )

    self._assets_db = db

    ############################################################################
    # Setup HDRI textures.
    if config.background_sampling_strategy == BackgroundSamplingStrategy.HDRI:
      self._hdri_source = kb.AssetSource.from_manifest(
          self._config.hdri_manifest
      )
      self._kubasic_source = kb.AssetSource.from_manifest(
          self._config.kubasic_manifest
      )
      if self._config.split_backgrounds:
        train_hdri, test_hdri = self._hdri_source.get_test_split(fraction=0.1)
        self._hdris = {
            "train": train_hdri,
            "test": test_hdri,
        }
      else:
        # Use all backgrounds for both splits.
        self._hdris = {
            "train": self._hdri_source.all_asset_ids,
            "test": self._hdri_source.all_asset_ids,
        }

  def __del__(self):
    kb.done()
    if os.path.exists(self._scratch_dir):
      logging.info("Removing scratch: %s", self._scratch_dir)
      shutil.rmtree(self._scratch_dir)

  def _setup_renderer(self):
    """Setup the renderer."""
    self._scene = kb.Scene.from_flags(self._config)
    self._simulator = kubric.simulator.pybullet.PyBullet(
        self._scene, self._scratch_dir
    )
    self._renderer = blender.Blender(
        self._scene,
        self._scratch_dir,
        # TODO(b/234817581): Issue in blender prevents using denoising.
        use_denoising=False,
        # Removes salt-and-pepper artifacts in shadows.
        adaptive_sampling=False,
    )

  def sample_point_in_sphere_shell_cap(
      self,
      inner_radius: float,
      outer_radius: float,
      min_height: float,
      reference_point: Optional[Tuple[float, float, float]] = None,
      min_distance: Optional[float] = None,
      max_distance: Optional[float] = None,
  ) -> Tuple[float, float, float]:
    """Sample in a given distance range from the origin and z >= min_height.

    If a reference_point is provided, the new point will be sampled minimally
    min_distance and maximimally max_distance away from the reference point.

    # NOTE: Random sampling may be inefficient for constraint-satisfaction when
    # max_distance - min_distance tends toward zero.

    Args:
      inner_radius: The radius of the inner sphere.
      outer_radius: The radius of the outer sphere.
      min_height: The minimum camera height.
      reference_point: A reference point for sampling nearby new points.
      min_distance: The minimum distance to the reference point.
      max_distance: The maximum distance to the reference point.

    Returns:
      A new point encoded as (x, y, z) coordinates.
    """
    assert self._cam_rng
    while True:
      v = self._cam_rng.uniform(
          (-outer_radius, -outer_radius, min_height),
          (outer_radius, outer_radius, outer_radius),
      )
      len_v = np.linalg.norm(v)
      if inner_radius <= len_v <= outer_radius:
        if reference_point is not None:
          distance_v = (
              np.linalg.norm(v - np.array(reference_point)))
          if min_distance <= distance_v <= max_distance:
            return tuple(v)
        else:
          return tuple(v)

  def _setup_camera(self):
    """Setup the scene camera."""
    assert self._cam_rng
    logging.info("Setting up the Camera...")
    self._scene.camera = kb.PerspectiveCamera(
        focal_length=self._config.focal_length,
        sensor_width=self._config.sensor_width,
    )
    num_frames = self._scene.frame_end - self._scene.frame_start + 1
    strategy = self._config.camera_sampling_strategy
    if strategy == CameraSamplingStrategy.SPHERE_SHELL:
      camera_positions = []
      for i in range(num_frames):
        camera_positions.append(
            self.sample_point_in_sphere_shell_cap(
                inner_radius=8.0,
                outer_radius=12.0,
                min_height=self._config.min_camera_height,
            )
        )

    elif strategy == CameraSamplingStrategy.SPHERE_SHELL_MIXED_DISTANCE:
      camera_positions = []
      for i in range(num_frames):
        sample_close = self._cam_rng.uniform() <= 0.5
        if i > 0 and sample_close:
          reference_camera_id = self._cam_rng.randint(len(camera_positions))
          distance_reference_point = camera_positions[reference_camera_id]
        else:
          distance_reference_point = None

        camera_positions.append(
            self.sample_point_in_sphere_shell_cap(
                inner_radius=8.0,
                outer_radius=12.0,
                min_height=self._config.min_camera_height,
                reference_point=distance_reference_point,
                min_distance=0.5,
                max_distance=1.5,
            )
        )

    elif strategy == CameraSamplingStrategy.CIRCLE:
      # height of the circle, only used for CameraSamplingStrategy.CIRCLE
      height = self._cam_rng.uniform(self._config.min_camera_height, 10)
      radius = 11
      l = np.sqrt(radius**2 - height**2)
      alpha = np.linspace(-np.pi, np.pi, num=200)
      x = np.cos(alpha) * l
      y = np.sin(alpha) * l
      z = np.ones_like(x) * height
      camera_positions = [(x[i], y[i], z[i]) for i in range(num_frames)]
    else:
      raise ValueError(f"Invalid CameraSamplingStrategy {strategy}")

    for pos, frame in zip(
        camera_positions,
        range(self._scene.frame_start, self._scene.frame_end + 1),
    ):
      self._scene.camera.position = pos
      self._scene.camera.look_at((0, 0, 0))
      self._scene.camera.keyframe_insert("position", frame)
      self._scene.camera.keyframe_insert("quaternion", frame)

  def _sample_asset_id(self, split: str) -> int:
    """Sample an asset id.

    Args:
      split: Dataset split. "train" or "test".

    Returns:
      An integer corresponding to an asset id.
    """
    assert self._obj_rng
    # Restrict view to assets with the desired dataset split.
    db = self._assets_db[self._assets_db["split"] == split]

    if self._config.object_sampling_strategy == ObjectSamplingStrategy.UNIFORM:
      # Sample an asset uniformly at random.
      return self._obj_rng.choice(db["id"])

    elif (
        self._config.object_sampling_strategy
        == ObjectSamplingStrategy.STRATIFIED_BY_CATEGORY
    ):
      # Sample a category at random, then an asset from the category.
      #
      # Sample a category ID.
      category_ids = db["category_id"].unique()
      category_id = self._obj_rng.choice(category_ids)

      # Sample an asset with that category id.
      eligible_assets = db[db["category_id"] == category_id]["id"]

      return self._obj_rng.choice(list(sorted(eligible_assets)))
    else:
      raise ValueError(
          "Unrecognized object_sampling_strategy: "
          f"{self._config.object_sampling_strategy}"
      )

  # --- Place random objects
  def add_random_object(self, spawn_region, split: str) -> kb.Object3D:
    """Add a random object at a random location and pose."""
    obj = self._asset_source.create(asset_id=self._sample_asset_id(split))

    # normalize the scale of the object such that one of the dimensions touches
    # the [-1, 1]^3 cube.
    #
    # Note: it is imperative that scale normalization happen before position
    # normalization.
    _normalize_object_scale(obj)
    _normalize_object_position(obj)
    assert np.all(np.abs(obj.aabbox) <= 1.0)

    # Get category id.
    category_id = int(obj.metadata["category_id"])
    categories = self._segmentation_ids()

    # Note that we add +1 when building the segmentation_id. This is because
    # the background has segmentation_id = 0.
    obj.segmentation_id = categories.index(category_id) + 1
    obj.metadata = {
        "asset_id": obj.asset_id,
        "category": obj.segmentation_id,
    }
    self._scene.add(obj)

    try:
      if (
          self._config.object_placement_sampling_strategy
          == ObjectPlacementSamplingStrategy.UNIFORM
      ):
        kb.move_until_no_overlap(
            obj, self._simulator, spawn_region=spawn_region, rng=self._obj_rng
        )
      elif (
          self._config.object_placement_sampling_strategy
          == ObjectPlacementSamplingStrategy.UPRIGHT
      ):
        _place_upright_object(obj, self._simulator, spawn_region, self._obj_rng)
      else:
        raise NotImplementedError(
            self._config.object_placement_sampling_strategy
        )
    except RuntimeError as e:
      logging.info("    Failed to place asset=%s", obj.asset_id)
      if e.args[0] == "Failed to place":
        self._scene.remove(obj)
      raise e

    logging.info(
        "    Succesfully placed asset=%s at position=%s",
        obj.asset_id,
        obj.position,
    )
    return obj

  def _segmentation_ids(self):
    """Generates a sorted list of semantic category ids."""
    return list(self._segmentation_info()["category_id"])

  def segmentation_labels(self) -> List[str]:
    """Generates a sorted list of semantic category names."""
    return list(self._segmentation_info()["category"])

  def _segmentation_info(self) -> pd.DataFrame:
    """Generates a DataFrame with semantic category ids and names."""
    df = self._assets_db
    df = df[["category_id", "category"]].drop_duplicates()
    df = df.sort_values("category_id")
    return df

  def try_setup_objects(self, split: str) -> bool:
    """Tries to setup the scene objects. Returns True on success."""
    assert self._obj_rng
    # Place random objects. If placement of any object fails, remove all
    # objects from the scene.
    num_static_objects = self._obj_rng.randint(
        self._config.min_num_objects, self._config.max_num_objects + 1
    )
    logging.info("Randomly placing %d static objects:", num_static_objects)
    objects = []
    for _ in range(num_static_objects):
      try:
        obj = self.add_random_object(
            spawn_region=self._config.spawn_region, split=split
        )
        objects.append(obj)
      except RuntimeError as e:
        logging.info(
            (
                "Object placement failed for at least 1 of %d objects. Removing"
                " all %d newly-added objects from the scene now."
            ),
            num_static_objects,
            len(objects),
        )
        for obj in objects:
          self._scene.remove(obj)
        if e.args[0] == "Failed to place":
          return False
        else:
          logging.info("Unexpected error when placing objects: %s", e)
          raise e
    logging.info("All %d objects added succesfully.", num_static_objects)
    return True

  def _add_walls(self):
    assert not self._walls
    logging.info("Creating walls...")

    # Add walls to enforce scene boundaries during simulation (removed later).
    #
    # The wall sizes are chosen to be 2x larger than the scene itself to
    # ensure that there are no gaps when running the physics simulation.
    x_min, y_min, z_min = self._config.spawn_region[0]
    x_max, y_max, z_max = self._config.spawn_region[1]
    dx, dy, dz = (x_max - x_min), (y_max - y_min), (z_max - z_min)
    x, y, z = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

    # The extra +1 and -1 offsets are because a default box has thickness 2
    # in each direction. The offsets shift the box over by half of the box's
    # thickness.
    walls = [
        {"scale": (dx, 1, dz), "position": (x, y_max + 1, z)},
        {"scale": (dx, 1, dz), "position": (x, y_min - 1, z)},
        {"scale": (1, dy, dz), "position": (x_max + 1, y, z)},
        {"scale": (1, dy, dz), "position": (x_min - 1, y, z)},
    ]
    del x_min, y_min, z_min
    del x_max, y_max, z_max
    del x, y, z, dx, dy, dz

    self._walls = []
    for i, w in enumerate(walls):
      w = kb.Cube(
          name=f"wall_{i}",
          scale=w["scale"],
          position=w["position"],
          static=True,
          background=True,
      )
      self._scene.add(w)
      self._walls.append(w)

  def _remove_walls(self):
    for w in self._walls:
      self._scene.remove(w)
    self._walls = []

  def _setup_background(self, split: str) -> str:
    """Setup scene's background."""
    assert self._bg_rng
    if (
        self._config.background_sampling_strategy
        == BackgroundSamplingStrategy.HDRI
    ):
      # Set an HDRI background.
      logging.info(
          "Choosing one of the %d backgrounds...", len(self._hdris[split])
      )
      background_hdri = self._hdri_source.create(
          asset_id=self._bg_rng.choice(self._hdris[split])
      )
      kb.assets.utils.add_hdri_dome(
          self._kubasic_source, self._scene, background_hdri=background_hdri
      )

      # Use HDRI for lighting.
      self._renderer._set_ambient_light_hdri(background_hdri.filename)  # pylint: disable=protected-access

      return kb.as_path(background_hdri.filename).stem

    elif (
        self._config.background_sampling_strategy
        == BackgroundSamplingStrategy.WHITE
    ):
      # Set background color = white
      self._scene.background = kb.get_color("white")

      # Add lights. Without an HDRI, we need to create our own lighting.
      sun = kb.DirectionalLight(
          color=kb.get_color("white"),
          shadow_softness=0.2,
          intensity=3.0,
          position=(11.6608, -6.62799, 25.8232),
      )
      lamp_back = kb.RectAreaLight(
          color=kb.get_color("white"),
          intensity=50.0,
          position=(-1.1685, 2.64602, 5.81574),
      )
      lamp_key = kb.RectAreaLight(
          color=kb.get_color(0xFFEDD0),
          intensity=100.0,
          width=0.5,
          height=0.5,
          position=(6.44671, -2.90517, 4.2584),
      )
      lamp_fill = kb.RectAreaLight(
          color=kb.get_color("#c2d0ff"),
          intensity=30.0,
          width=0.5,
          height=0.5,
          position=(-4.67112, -4.0136, 3.01122),
      )
      lights = [sun, lamp_back, lamp_key, lamp_fill]

      for light in lights:
        light.look_at((0, 0, 0))
        self._scene.add(light)

      return "white"

    else:
      raise NotImplementedError(
          "background_sampling_strategy = "
          f"{self._config.background_sampling_strategy} is not implemented."
      )

  def _add_floor(self) -> kb.Cube:
    """Create a flat, thin cube to be the floor."""
    # Choose floor material.
    if (
        self._config.background_sampling_strategy
        == BackgroundSamplingStrategy.WHITE
    ):
      floor_material = kb.PrincipledBSDFMaterial(
          color=kb.get_color("white"), roughness=1.0, specular=0.0
      )
    elif (
        self._config.background_sampling_strategy
        == BackgroundSamplingStrategy.HDRI
    ):
      floor_material = kb.UndefinedMaterial()
    else:
      raise NotImplementedError(self._config.background_sampling_strategy)

    # Add floor.
    floor = kb.Cube(
        name="floor",
        scale=(100, 100, 1),
        position=(0, 0, -1),
        friction=0.5,
        restitution=0,
        material=floor_material,
        static=True,
        background=True,
    )
    self._scene.add(floor)
    return floor

  def render_scene(self, seed: int, split: str) -> Dict[str, Any]:
    """Creates and renders a scene."""
    cam_seed, obj_seed, bg_seed = np.random.SeedSequence(seed).generate_state(3)
    if self._config.fix_cam_seed is not None:
      cam_seed = self._config.fix_cam_seed
    if self._config.fix_obj_seed is not None:
      obj_seed = self._config.fix_obj_seed
    if self._config.fix_bg_seed is not None:
      bg_seed = self._config.fix_bg_seed

    self._cam_rng = np.random.RandomState(seed=cam_seed)
    self._obj_rng = np.random.RandomState(seed=obj_seed)
    self._bg_rng = np.random.RandomState(seed=bg_seed)

    background_name = self._setup_background(split=split)
    logging.info("Setting background to %s", background_name)

    self._setup_camera()

    logging.info("Setting up objects...")
    # Note: we place objects BEFORE defining the floor and walls. This ensures
    # that PyBullet only checks object-to-object collisions during placement.
    while True:
      if self.try_setup_objects(split=split):
        break

    logging.info("Creating floor...")
    floor = self._add_floor()

    # --- Simulation
    if self._config.num_steps_simulation:
      logging.info(
          "Running %d frames of simulation to let objects settle.",
          self._config.num_steps_simulation,
      )
      self._add_walls()
      _, _ = self._simulator.run(
          frame_start=-1 * self._config.num_steps_simulation, frame_end=0
      )
      self._remove_walls()

    # --- Ensure objects lie in desired bounding box.
    scene_boundaries = _get_scene_boundaries(
        floor=floor,
        spawn_region=self._config.spawn_region,
        foreground_objects=list(self._scene.foreground_assets),
    )

    # --- Rendering
    logging.info("Rendering the scene ...")
    render_data = self._renderer.render(
        return_layers=["rgba", "segmentation", "depth"]
    )

    render_data["instances"] = render_data["segmentation"]

    # Warning: this overwrites the segmentation field with object categories.
    render_data["segmentation"] = kb.adjust_segmentation_idxs(
        render_data["segmentation"],
        self._scene.assets,
        self._scene.foreground_assets,
    )

    logging.info("Collating metadata.")
    scene_metadata = kb.get_scene_metadata(self._scene, seed=seed)
    scene_camera = kb.get_camera_info(
        self._scene.camera,
        height=scene_metadata["resolution"][1],
        width=scene_metadata["resolution"][0],
    )
    scene_camera["positions"] = scene_camera["positions"].tolist()
    scene_camera["quaternions"] = scene_camera["quaternions"].tolist()
    scene_camera["K"] = scene_camera["K"].tolist()
    scene_camera["R"] = scene_camera["R"].tolist()

    results = {
        "render_data": render_data,
        "metadata": scene_metadata,
        "camera": scene_camera,
        "scene_boundaries": scene_boundaries,
        "instances": kb.get_instance_info(self._scene),
        "background": {
            "background_name": background_name,
        },
        "segmentation_labels": self.segmentation_labels(),
    }

    # Cleanup
    logging.info("Deleting %d exiting assets.", len(self._scene.assets))
    for asset in self._scene.assets:
      self._scene.remove(asset)
    logging.info(
        "try_setup_objects: scene has %d assets.", len(self._scene.assets)
    )
    return results
