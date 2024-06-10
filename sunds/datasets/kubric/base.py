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

"""Abstract base for renderer and config."""

import abc
from typing import Any, Dict, List


class IrrecoverableRenderingError(Exception):
  """Raise to terminate rendering and continue (NOTE: this causes data loss)."""
  pass


class SceneConfig(abc.ABC):
  object_types: Any
  frame_end: int = 1


class BaseRenderer(abc.ABC):
  """Renderer for a Kubric scene."""

  @abc.abstractmethod
  def __init__(self, config: SceneConfig):
    ...

  @abc.abstractmethod
  def render_scene(self, seed: int, split: str) -> Dict[str, Any]:
    ...

  @abc.abstractmethod
  def segmentation_labels(self) -> List[str]:
    ...
