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

"""Wrapper for tdqm."""

from typing import TypeVar

from absl import logging
from sunds.utils import py_utils
import tqdm as tqdm_base


_IterableT = TypeVar('_IterableT')


class _LogFile:
  """A File-like object that log to INFO."""

  def write(self, message):
    logging.info(message)

  def flush(self):
    pass


def tqdm(iterable: _IterableT, **kwargs) -> _IterableT:
  """Add a progressbar to the iterable."""
  return tqdm_base.tqdm(iterable, **kwargs)
