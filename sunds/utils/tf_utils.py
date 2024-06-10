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

"""Tensorflow utils."""

import functools
import operator
from typing import Sequence, Tuple, Union

import numpy as np
from sunds.typing import Shape, Tensor, TensorLike  # pylint: disable=g-multiple-import
import tensorflow as tf

_MinMaxValue = Union[int, float, np.ndarray, Sequence[int], Sequence[float]]


def interp(
    x: Tensor,
    from_: Tuple[_MinMaxValue, _MinMaxValue],
    to: Tuple[_MinMaxValue, _MinMaxValue],
    axis: int = -1,
) -> Tensor:
  """Linearly scale the given value by the given range.

  Somehow similar to `np.interp` or `scipy.interpolate.inter1d` with some
  differences like support scaling an axis by a different factors and
  extrapolate values outside the boundaries.

  `from_` and `to` are expected to be `(min, max)` tuples and the function
  interpolate between the two ranges.

  Example: Normalizing a uint8 image to `(-1, 1)`.

  ```python
  img = jnp.array([
      [0, 0],
      [127, 255],
  ])
  img = j3d.interp(img, (0, 255), (0, 1))
  img == jnp.array([
      [-1, -1],
      [0.498..., 1],
  ])
  ```

  `min` and `max` can be either float values or array like structure, in which
  case the numpy broadcasting rules applies (x should be a `Array[... d]` and
  min/max values should be `Array[d]`.

  Example: Converting normalized 3d coordinates to world coordinates.

  `coords[:, 0]` is interpolated from `(0, h)` to `(-1, 1)` and `coords[:, 1]`
  from `(0, w)` to `(-1, 1)`,...

  ```python
  coords = j3d.interp(coords, from_=(-1, 1), to=(0, (h, w, d)), to=(-1, 1))
  ```

  * `coords[:, 0]` is interpolated from `(-1, 1)` to `(0, h)`
  * `coords[:, 1]` is interpolated from `(-1, 1)` to `(0, w)`
  * `coords[:, 2]` is interpolated from `(-1, 1)` to `(0, d)`

  Args:
    x: Array to scale
    from_: Range of x.
    to: Range to which normalize x.
    axis: Axis on which normalizing. Only relevant if `from_` or `to` items
      contains range value.

  Returns:
    Float tensor with same shape as x, but with normalized coordinates.
  """
  # Could add an `axis` argument.
  # Could add an `fill_values` argument to indicates the behavior if input
  # values are outside the input range. (`error`, `extrapolate` or `truncate`).

  if x.dtype != tf.float32:
    raise ValueError(f'interp input should be float32. Got: {x.dtype}')

  if axis != -1:
    raise NotImplementedError(
        'Only last axis supported for now. Please send a feature request.'
    )

  # Normalize to array (to support broadcasting).
  # If inputs are static arguments (not `tf.Tensor`), use numpy arrays for
  # optimization (factors statically computed).
  from_ = tuple(_to_array(v) for v in from_)
  to = tuple(_to_array(v) for v in to)

  # `a` can be scalar or array of shape=(x.shape[-1],), same for `b`
  a, b = _linear_interp_factors(*from_, *to)
  return a * x + b


def _linear_interp_factors(
    old_min: _MinMaxValue,
    old_max: _MinMaxValue,
    new_min: _MinMaxValue,
    new_max: _MinMaxValue,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
  """Resolve the `y = a * x + b` equation and returns the factors."""
  a = (new_min - new_max) / (old_min - old_max)
  b = (old_min * new_max - new_min * old_max) / (old_min - old_max)
  return a, b


def random_choice(
    a: Union[int, TensorLike],
    *,
    size: Union[None, int, Shape] = None,
    replace: bool = True,
) -> tf.Tensor:
  """TF implementation of `np.random.Generator.choice`.

  Args:
    a: Tensor to which select samples. If an `int`, the random sample is
      generated from `tf.range(a)`.
    size: Number of sample to select (can be `int` or output shape)
    replace: Whether the sample is with or without replacement. Default is
      `True`, meaning that `a` value of a can be selected multiple times.

  Returns:
    samples: The choices sampled from a.
  """
  if isinstance(a, int):  # Normalize int
    a = tf.range(a)
  a = tf.convert_to_tensor(a)
  if replace:
    raise NotImplementedError(
        'Only replace=False supported. You can use `tf.random.categorical.`'
    )
  if len(a.shape) != 1:
    raise NotImplementedError(
        '`random_choice` only support single dim tensors.'
    )

  shape = (size,) if isinstance(size, int) else tuple(size)
  # TODO(py3.8): Replace by `math.prod`
  num_samples = functools.reduce(operator.mul, shape)
  num_values = a.shape[0]  # Numbe of values to sample
  if not replace and num_samples > num_values:
    raise ValueError(
        'Cannot take a larger sample than population when `replace=False`. '
        f'{num_samples} > {num_values}'
    )

  indices = tf.range(num_values)
  indices = tf.random.shuffle(indices)
  indices = indices[:num_samples]

  a = tf.gather(a, indices)
  return tf.reshape(a, shape)


def _to_array(x: _MinMaxValue) -> _MinMaxValue:
  """Convert to array/tensor."""
  if isinstance(x, tf.Tensor):
    assert x.dtype == tf.float32, x.dtype
    return x
  else:
    return np.array(x)
