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

"""Path utils."""

import functools
import typing

import tensorflow_datasets as tfds

# TODO(epot): Should move the pathlib-abstraction to a self-contained
# library
# pathlib-like abstraction
Path = tfds.core.ReadWritePath

# Convert a resource path to write path.
# Used for automated scripts which write to sunds/
write_path = tfds.core.utils.to_write_path


@functools.lru_cache()
def sunds_dir() -> Path:
  """Root directory for `sunds/`."""
  path = tfds.core.utils.resource_path('sunds')
  return typing.cast(Path, path)
