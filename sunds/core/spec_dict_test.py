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

"""Tests for spec_dict."""

from typing import Any, NamedTuple

import pytest
from sunds.core import spec_dict
import tensorflow as tf
import tensorflow_datasets as tfds


class _Param(NamedTuple):
  """`pytest.mark.parametrize` single param."""

  value: Any
  default: Any
  expected: Any


@pytest.mark.parametrize(
    'value, default, expected',
    [
        _Param(False, tfds.features.Image(), {}),
        _Param(
            value=True,  # True: use default
            default=tfds.features.Image(),
            expected={'a': tfds.features.Image()},
        ),
        _Param(
            value=True,
            default=tf.int32,  # Also works with dtype
            expected={
                'a': tf.int32,
            },
        ),
        _Param(
            value=tfds.features.Text(),  # Feature give, use it
            default=tfds.features.Image(),
            expected={'a': tfds.features.Text()},
        ),
        _Param(
            value=['cat', 'dog'],  # Label to use
            default=spec_dict.labeled_image(shape=(28, 28, 1)),
            expected={
                'a': tfds.features.LabeledImage(
                    labels=['cat', 'dog'],
                    shape=(28, 28, 1),
                ),
            },
        ),
        _Param(
            value=True,  # Use default, but without explicit labels
            default=spec_dict.labeled_image(shape=(28, 28, 1)),
            expected={
                'a': tfds.features.LabeledImage(
                    labels=None,
                    shape=(28, 28, 1),
                ),
            },
        ),
        # Test nested case
        _Param(
            value=True,
            default={'x': tf.int64, 'y': tfds.features.Text()},
            expected={
                'a': {'x': tf.int64, 'y': tfds.features.Text()},
            },
        ),
        _Param(
            value={'x': tf.int32, 'y': tfds.features.Text(), 'z': tf.string},
            default={'x': tf.int64, 'y': tfds.features.Text()},
            expected={
                'a': {'x': tf.int32, 'y': tfds.features.Text(), 'z': tf.string},
            },
        ),
    ],
)
def test_spec_dict_maybe_set(value, default, expected):
  spec = spec_dict.SpecDict()
  spec.maybe_set('a', value, default)
  tfds.testing.assert_features_equal(spec, expected)


@pytest.mark.parametrize(
    'other, default, expected',
    [
        _Param(
            value=False,
            default={'x': tfds.features.Image()},
            expected={},
        ),
        _Param(
            value=True,
            default={'x': tfds.features.Image()},
            expected={'x': tfds.features.Image()},
        ),
        _Param(
            value=True,
            default={'x': tf.int64},
            expected={'x': tf.int64},
        ),
        _Param(
            value={'x': tfds.features.Text(), 'y': tf.float32},
            default={'x': tfds.features.Image()},
            expected={'x': tfds.features.Text(), 'y': tf.float32},
        ),
    ],
)
def test_spec_dict_maybe_update(other, default, expected):
  spec = spec_dict.SpecDict()
  spec.maybe_update(other, default)
  tfds.testing.assert_features_equal(spec, expected)
