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

"""Tests for tf_utils."""

import functools

import pytest
from sunds import utils
import tensorflow as tf

_tf_const = functools.partial(tf.constant, dtype=tf.float32)


def test_interp_scalar():

  vals = _tf_const([
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0.5, 1],
      [1, 1],
  ])
  tf.debugging.assert_near(
      utils.interp(vals, from_=(-1, 1), to=(0, 256)),
      [
          [0, 0],
          [0, 128],
          [0, 256],
          [192, 256],
          [256, 256],
      ],
  )
  tf.debugging.assert_near(
      utils.interp(vals, from_=(-1, 1), to=(0, 1)),
      [
          [0, 0],
          [0, 0.5],
          [0, 1],
          [0.75, 1],
          [1, 1],
      ],
  )

  vals = _tf_const([
      [255, 255, 0],
      [255, 128, 0],
      [255, 0, 128],
  ])
  tf.debugging.assert_near(
      utils.interp(vals, from_=(0, 255), to=(0, 1)),
      [
          [1, 1, 0],
          [1, 128 / 255, 0],
          [1, 0, 128 / 255],
      ],
  )
  tf.debugging.assert_near(
      utils.interp(vals, from_=(0, 255), to=(-1, 1)),
      [
          [1, 1, -1],
          [1, 0.00392163, -1],
          [1, -1, 0.00392163],
      ],
  )


def test_interp_coords():

  coords = _tf_const([
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0.5, 1],
      [1, 1],
  ])
  tf.debugging.assert_near(
      utils.interp(coords, (-1, 1), (0, (1024, 256))),
      [
          [0, 0],
          [0, 128],
          [0, 256],
          [768, 256],
          [1024, 256],
      ],
  )

  coords = _tf_const([
      [[0, 0], [0, 1024]],
      [[256, 256], [0, 768]],
  ])
  tf.debugging.assert_near(
      utils.interp(coords, (0, (256, 1024)), (0, 1)),
      [
          [[0, 0], [0, 1]],
          [[1, 0.25], [0, 0.75]],
      ],
  )


def test_random_choice():
  # Custom values selected
  y = utils.random_choice([0, 10, 20], size=2, replace=False).numpy()
  assert y.shape == (2,)
  assert not set(y) - {0, 10, 20}
  assert len(set(y)) == len(y)  # All elements are distinct

  # Try many values to make sure each value is only selected once
  y = utils.random_choice(list(range(100)), size=50, replace=False).numpy()
  assert y.shape == (50,)
  assert not set(y) - set(range(100))
  assert len(set(y)) == 50

  # Test with `int`
  y = utils.random_choice(100, size=50, replace=False).numpy()
  assert y.shape == (50,)
  assert not set(y) - set(range(100))
  assert len(set(y)) == 50

  # Test with `int` & complement
  y, y_c = utils.random_choice(
      100, size=45, replace=False, return_complement=True)
  y = y.numpy()
  y_c = y_c.numpy()
  assert y.shape == (45,)
  assert y_c.shape == (55,)
  assert set(y).union(set(y_c)) == set(range(100))
  assert set(y).isdisjoint(set(y_c))
  assert not set(y) - set(range(100))
  assert not set(y_c) - set(range(100))

  # Test with shape
  y = utils.random_choice(list(range(10)), size=(2, 3), replace=False).numpy()
  assert y.shape == (2, 3)
  assert not set(y.flatten()) - set(range(100))
  assert len(set(y.flatten())) == 6

  with pytest.raises(ValueError, match='larger sample than population'):
    utils.random_choice([1, 2, 3], size=(2, 3), replace=False)
