# Copyright 2023 The sunds Authors.
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

"""Typing annotations of sunds."""
# This file is at the top-level folder to allow `from sunds.typing import xyz`

from typing import Any, Iterable, List, Union

from etils import epath
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ==== Structural types ====

Tree = tfds.typing.Tree
TreeDict = tfds.typing.TreeDict

Json = tfds.typing.Json
JsonValue = tfds.typing.JsonValue

# ==== Tensor types ====

Dim = tfds.typing.Dim
Shape = tfds.typing.Shape

Tensor = tfds.typing.Tensor
# Nested dict of tensor
TensorDict = TreeDict[Tensor]
# TensorLike accept any compatible tensor values (np.array, list,...)
TensorLike = Union[Tensor, Any]

Array = np.ndarray
ArrayDict = TreeDict[Array]
ArrayLike = Union[Array, Any]

# ==== TFDS types ====
# TODO(epot): Could this be replaced by some tfds.typing.Xyz

Split = Union[str, tfds.core.Split, tfds.core.ReadInstruction]

_FeatureSpecElem = Union[tfds.features.FeatureConnector, tf.dtypes.DType]
FeatureSpecs = TreeDict[_FeatureSpecElem]
# Hint to specify features specs. E.g.
# {'color_images'}
# {'color_images': True}
# {'color_images': tfds.features.Image()}
_FeatureSpecHintElem = Union[_FeatureSpecElem, Iterable[str], bool]
FeatureSpecsHint = TreeDict[_FeatureSpecHintElem]

# Args accepted by `tfds.features.LabeledImage`:
# * `['background', 'car']`: List of label names
# * `/path/to/labels.txt`: Path to label files (one label per line)
# * `int`: Number of labels (if label strings unknown)
# * `None`: if label info unknown
LabelArg = Union[List[str], epath.PathLike, int, None]

FeaturesOrBool = Union[FeatureSpecs, bool]
LabelOrFeaturesOrBool = Union[LabelArg, FeaturesOrBool]
