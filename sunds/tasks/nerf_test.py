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

"""Tests for nerf."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import sunds
from sunds.tasks import boundaries_utils
from sunds.typing import FeatureSpecsHint
import tensorflow as tf
import tensorflow_datasets as tfds


@pytest.mark.parametrize('add_name', [False, True], ids=['noname', 'name'])
def test_nerf(lego_builder: sunds.core.DatasetBuilder, add_name: bool):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(add_name=add_name),
  )
  assert isinstance(ds, tf.data.Dataset)
  img_shape = (800, 800)
  spec = {
      'ray_directions': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
      'ray_origins': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
      'color_image': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.uint8),
  }
  if add_name:
    spec.update({
        'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'camera_name': tf.TensorSpec(shape=(), dtype=tf.string),
    })
  assert ds.element_spec == spec
  list(ds)  # Pipeline can be executed


def test_nerf_frame_only(lego_builder_frame_only: sunds.core.DatasetBuilder):
  data_dir = lego_builder_frame_only.data_dir
  builder = sunds.builder('nerf_synthetic/lego', data_dir=data_dir)
  ds = builder.as_dataset(split='train', task=sunds.tasks.Nerf())
  assert isinstance(ds, tf.data.Dataset)

  with pytest.raises(tfds.core.DatasetNotFoundError):
    _ = builder.scene_builder.info  # Loading scene raise error


@pytest.mark.parametrize('add_name', [False, True], ids=['noname', 'name'])
def test_nerf_tfds_id(lego_builder: sunds.core.DatasetBuilder, add_name: bool):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(add_name=add_name),
      read_config=tfds.ReadConfig(add_tfds_id=True),
  )
  assert isinstance(ds, tf.data.Dataset)
  img_shape = (800, 800)
  spec = {
      'ray_directions': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
      'ray_origins': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.float32),
      'color_image': tf.TensorSpec(shape=(*img_shape, 3), dtype=tf.uint8),
      'tfds_id': tf.TensorSpec(shape=(), dtype=tf.string),
  }
  if add_name:
    spec.update({
        'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'camera_name': tf.TensorSpec(shape=(), dtype=tf.string),
    })

  assert ds.element_spec == spec
  list(ds)  # Pipeline can be executed


@pytest.mark.parametrize('add_name', [False, True], ids=['noname', 'name'])
@pytest.mark.parametrize(
    'yield_mode, batch_shape',
    [
        ('ray', ()),
        ('image', (800, 800)),
        ('stacked', (1, 800, 800)),
    ],
)
def test_nerf_batch_shape(
    lego_builder: sunds.core.DatasetBuilder,
    yield_mode: str,
    batch_shape: tuple[int, ...],
    add_name: bool,
):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(yield_mode=yield_mode, add_name=add_name),
  )
  assert isinstance(ds, tf.data.Dataset)
  camera_shape = (1,) if yield_mode == 'stacked' else ()
  spec = {
      'ray_directions':
          tf.TensorSpec(shape=(*batch_shape, 3), dtype=tf.float32),
      'ray_origins':
          tf.TensorSpec(shape=(*batch_shape, 3), dtype=tf.float32),
      'color_image':
          tf.TensorSpec(shape=(*batch_shape, 3), dtype=tf.uint8),
  }
  if add_name:
    spec.update({
        'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'camera_name': tf.TensorSpec(shape=camera_shape, dtype=tf.string),
    })
  assert ds.element_spec == spec
  list(ds.take(4))  # Pipeline can be executed


@pytest.mark.parametrize('add_name', [False, True], ids=['noname', 'name'])
def test_nerf_yield_all_camera(
    lego_builder: sunds.core.DatasetBuilder,
    add_name: bool,
):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(yield_mode='dict', add_name=add_name),
  )
  assert isinstance(ds, tf.data.Dataset)
  img_shape = (800, 800)
  spec = {
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
  }
  if add_name:
    spec.update({
        'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
        'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
    })
  assert ds.element_spec == spec
  list(ds)  # Pipeline can be executed


def test_nerf_normalize_rays(lego_builder: sunds.core.DatasetBuilder):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(normalize_rays=True, add_name=True),
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


@pytest.mark.parametrize('yield_mode', ['image', 'stacked', 'dict'])
def test_nerf_additional_specs(
    lego_builder: sunds.core.DatasetBuilder,
    yield_mode: str,
):
  ds = lego_builder.as_dataset(
      split='train',
      task=sunds.tasks.Nerf(
          additional_frame_specs={'timestamp', 'pose'},
          additional_camera_specs={'intrinsics'},
          yield_mode=yield_mode,
      ),
  )
  assert 'timestamp' in ds.element_spec
  assert 'pose' in ds.element_spec
  if yield_mode == 'dict':
    assert 'intrinsics' in ds.element_spec['cameras']['default_camera']
  else:
    assert 'intrinsics' in ds.element_spec
  assert len(ds)  # pylint: disable=g-explicit-length-test
  list(ds)  # Pipeline can be executed


@pytest.mark.parametrize('add_name', [False, True], ids=['noname', 'name'])
@pytest.mark.parametrize(
    'normalize_rays', [False, True], ids=['nonorm', 'norm'])
@pytest.mark.parametrize(
    'yield_mode',
    ['ray', 'image', 'stacked', 'dict'],
)
@pytest.mark.parametrize('additional_frame_specs', [{}, {'timestamp'}])
@pytest.mark.parametrize(
    'remove_invalid_rays', [False, True], ids=['keep', 'remove'])
def test_all_flags(
    lego_builder: sunds.core.DatasetBuilder,
    normalize_rays: bool,
    yield_mode: str,
    additional_frame_specs: FeatureSpecsHint,
    remove_invalid_rays: bool,
    add_name: bool,
):
  """Test which checks that all combinations of options work together."""
  # Some conbinaitions are incompatible
  if additional_frame_specs and yield_mode == 'ray':
    error_cm = pytest.raises(
        NotImplementedError,
        match='frame specs not compatible',
    )
  else:
    # TODO(py3.7): Replace by contextlib.nullcontext
    error_cm = contextlib.suppress()

  with error_cm:
    ds = lego_builder.as_dataset(
        split='train',
        task=sunds.tasks.Nerf(
            yield_mode=yield_mode,
            normalize_rays=normalize_rays,
            additional_frame_specs=additional_frame_specs,
            remove_invalid_rays=remove_invalid_rays,
            add_name=add_name,
        ),
    )
    if not remove_invalid_rays:
      assert len(ds)  # `len` is preserved  # pylint: disable=g-explicit-length-test
    list(ds.take(2))  # Pipeline can be executed


def test_center_example():
  ex = {
      'cameras': {
          'target': {
              'ray_origins':
                  tf.constant(
                      [
                          [0, 3, 7],
                          # The second ray has an invalid direction, so its
                          # origin should be ignored for the center calculation.
                          [1000, 1000, 1000],
                      ],
                      dtype=tf.float32),
              'ray_directions':
                  tf.constant([
                      [1, 0, 0],
                      [0, 0, 0],
                  ], dtype=tf.float32),
          },
          'input0': {
              'ray_origins': tf.constant([
                  [0, 3, 7],
              ], dtype=tf.float32),
              'ray_directions': tf.constant([
                  [0, 1, 0],
              ], dtype=tf.float32),
          },
      },
  }

  centered_ex = sunds.tasks.nerf._center_example(
      ex, far_plane_for_centering=10, jitter=0)

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
              'ray_origins': tf.constant([
                  [-5, -5, 0],
              ], dtype=tf.float32),
              'ray_directions': tf.constant([
                  [0, 1, 0],
              ], dtype=tf.float32),
          },
      },
  }

  tf.nest.map_structure(tf.debugging.assert_near, centered_ex, expected_ex)
