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

"""Helper custom types."""

import enum


class MetricGroupings(enum.Enum):
  """Semantic metric groupings."""
  MOST_IMPORTANT = "most_important"
  IMPORTANT = "important"
  MOVERS = "movers"
  GROUND = "ground"
  SMALL = "small"
  RARE = "rare"
  ATOMIC_MAPS = "atomic_maps"
  IMAGE_ONLY = "image_only"
  LIDAR_ONLY = "lidar_only"

  @classmethod
  def has_value(cls, value):
    return value in cls._value2member_map_.values()

  @classmethod
  def lidar_only(cls, value):
    return value == cls.LIDAR_ONLY
