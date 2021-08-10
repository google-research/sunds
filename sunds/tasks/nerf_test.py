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

"""Tests for nerf."""

import contextlib
from typing import Callable

import numpy as np
import pytest
import sunds
from sunds.tasks import boundaries_utils
from sunds.typing import FeatureSpecs
import tensorflow as tf


def test_nerf(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(split='train', task=sunds.tasks.Nerf())
  assert isinstance(ds, tf.data.Dataset)
  img_shape = (800, 800)
  assert ds.element_spec == {
      'ray_directions': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
      'ray_origins': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
      'color_image': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.uint8),
      'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
      'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
      'camera_name': tf.TensorSpec(shape=(), dtype=tf.string),
  }
  list(ds)  # Pipeline can be executed


def test_nerf_flatten_img(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(keep_as_image=False),
  )
  assert isinstance(ds, tf.data.Dataset)
  assert ds.element_spec == {
      'ray_directions': tf.TensorSpec(shape=(3,), dtype=tf.float32),
      'ray_origins': tf.TensorSpec(shape=(3,), dtype=tf.float32),
      'color_image': tf.TensorSpec(shape=(3,), dtype=tf.uint8),
      'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
      'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
      'camera_name': tf.TensorSpec(shape=(), dtype=tf.string),
  }
  list(ds.take(4))  # Pipeline can be executed


def test_nerf_yield_all_camera(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(yield_individual_camera=False),
  )
  assert isinstance(ds, tf.data.Dataset)
  img_shape = (800, 800)
  assert ds.element_spec == {
      'cameras': {
          'default_camera': {  # Default camera
              'ray_directions':
                  tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
              'ray_origins':
                  tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
              'color_image':
                  tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.uint8),
          },
      },
      'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
      'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
  }
  list(ds)  # Pipeline can be executed


def test_nerf_normalize_rays(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(normalize_rays=True),
  )
  list(ds)  # Pipeline can be executed


def test_normalize_rays():
  normalize_rays = sunds.tasks.nerf._normalize_rays

  ex = {
      'ray_origins':
          tf.constant([
              [0., -2., -2],
              [2., 2., 2.],
              [8., -2., 4.],
              [8., -2., 4.],
          ]),
      'ray_directions':
          tf.constant([
              [0., 0., 23.],
              [1., 1., 0.],
              [3., 3., 0.],
              [0., 0., 0.],
          ]),
  }
  scene_ex = [
      {
          'scene_name': 'scene_00',
          'scene_box': {
              'min_corner': np.array([-20, -20, -20], dtype=np.float32),
              'max_corner': np.array([20, 20, 20], dtype=np.float32),
          },
      },
      {
          'scene_name': 'scene_01',
          'scene_box': {
              'min_corner': np.array([0, -2, -2], dtype=np.float32),
              'max_corner': np.array([8, 2, 6], dtype=np.float32),
          },
      },
  ]
  scene_boundaries = boundaries_utils.MultiSceneBoundaries(scene_ex)
  normalized_ex = normalize_rays(
      {'scene_name': 'scene_01'},
      ex,
      scene_boundaries=scene_boundaries,
  )

  expected_ex = {
      'ray_origins':
          tf.constant([
              [-1., -1., -1.],
              [-0.5, 1., 0.],
              [1., -1., 0.5],
              [1., -1., 0.5],
          ]),
      'ray_directions':
          tf.constant([
              [0., 0., 1.],
              [0.4472136, 0.8944272, 0.],
              [0.4472136, 0.8944272, 0.],
              [0., 0., 0.],
          ]),
  }

  tf.nest.map_structure(tf.debugging.assert_near, normalized_ex, expected_ex)
  tf.debugging.assert_near(
      tf.norm(normalized_ex['ray_directions'], axis=-1),
      [1., 1., 1., 0.],
  )

  with pytest.raises(
      tf.errors.InvalidArgumentError, match='Unknown scene name'):
    normalized_ex = normalize_rays(
        {'scene_name': 'unknown_scene'},
        ex,
        scene_boundaries=scene_boundaries,
    )


def _scene_id_spec() -> FeatureSpecs:
  frame_spec = sunds.specs.frame_spec({})
  return {'frame_name': frame_spec['frame_name']}


@pytest.mark.parametrize('normalize_rays', [False, True])
@pytest.mark.parametrize('yield_individual_camera', [False, True])
def test_nerf_additional_specs(
    lego_builder: sunds.core.DatasetBuilder,
    yield_individual_camera: bool,
    normalize_rays: bool,
):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(
          additional_frame_specs=_scene_id_spec(),
          normalize_rays=normalize_rays,
          yield_individual_camera=yield_individual_camera,
      ),
  )
  assert (ds.element_spec['frame_name'] == tf.TensorSpec(
      shape=(), dtype=tf.string))
  list(ds.take(2))  # Pipeline can be executed


@pytest.mark.parametrize('normalize_rays', [False, True], ids=[None, 'norm'])
@pytest.mark.parametrize('keep_as_image', [False, True], ids=['ray', 'img'])
@pytest.mark.parametrize(
    'yield_individual_camera', [False, True], ids=['multi', 'single'])
@pytest.mark.parametrize('additional_frame_specs', [dict, _scene_id_spec])
@pytest.mark.parametrize(
    'remove_invalid_rays', [False, True], ids=['remove', None])
def test_all_flags(
    lego_builder: sunds.core.DatasetBuilder,
    normalize_rays: bool,
    keep_as_image: bool,
    yield_individual_camera: bool,
    additional_frame_specs: Callable[[], FeatureSpecs],
    remove_invalid_rays: bool,
):
  """Test which checks that all combinations of options work together."""
  additional_frame_specs = additional_frame_specs()

  # Some conbinaitions are incompatible
  if additional_frame_specs and not keep_as_image:
    error_cm = pytest.raises(ValueError, match='frame specs not compatible')
  else:
    # TODO(py3.7): Replace by contextlib.nullcontext
    error_cm = contextlib.suppress()

  with error_cm:
    ds = lego_builder.as_dataset(
        split='train',
        task=sunds.tasks.Nerf(
            keep_as_image=keep_as_image,
            normalize_rays=normalize_rays,
            yield_individual_camera=yield_individual_camera,
            additional_frame_specs=additional_frame_specs,
            remove_invalid_rays=remove_invalid_rays,
        ),
    )
    list(ds.take(2))  # Pipeline can be executed


def test_center_example():
  ex = {
      'cameras': {
          'target': {
              'ray_origins':
                  tf.constant([
                      [0, 3, 7],
                      # The second ray has an invalid direction, so its origin
                      # should be ignored for the center calculation.
                      [1000, 1000, 1000],
                  ], dtype=tf.float32),
              'ray_directions':
                  tf.constant([
                      [1, 0, 0],
                      [0, 0, 0],
                  ], dtype=tf.float32),
              },
          'input0': {
              'ray_origins':
                  tf.constant([
                      [0, 3, 7],
                  ], dtype=tf.float32),
              'ray_directions':
                  tf.constant([
                      [0, 1, 0],
                  ], dtype=tf.float32),
          },
      },
  }

  centered_ex = sunds.tasks.nerf._center_example(ex,
                                                 far_plane_for_centering=10,
                                                 jitter=0)

  expected_ex = {
      'cameras': {
          'target': {
              'ray_origins':
                  tf.constant([
                      [-5, -5, 0],
                      [995, 992, 993],
                  ], dtype=tf.float32),
              'ray_directions':
                  tf.constant([
                      [1, 0, 0],
                      [0, 0, 0],
                  ], dtype=tf.float32),
              },
          'input0': {
              'ray_origins':
                  tf.constant([
                      [-5, -5, 0],
                  ], dtype=tf.float32),
              'ray_directions':
                  tf.constant([
                      [0, 1, 0],
                  ], dtype=tf.float32),
          },
      },
  }

  tf.nest.map_structure(tf.debugging.assert_near, centered_ex, expected_ex)
