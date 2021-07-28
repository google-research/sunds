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

"""Util to define Spec dataclasses."""

import datetime
from typing import Any, Callable, Dict, Type, TypeVar

import dataclasses

_T = TypeVar('_T')


class SpecBase:
  """Base example structure."""

  def asdict(self) -> Dict[str, Any]:
    """Convert the dataclass to dict, recursivelly applied to childs."""
    attrs = {}
    for field in dataclasses.fields(self):
      value = getattr(self, field.name, None)
      if value is None:  # Filter None elements
        continue
      if _isregistered(value):
        value = _asdict(value)
      elif (  # Recurse on dicts
          isinstance(value, dict)
          and value
          and _isregistered(next(iter(value.values())))
      ):
        value = {k: _asdict(v) for k, v in value.items()}
      elif isinstance(value, list) and value and _isregistered(value[0]):
        value = [_asdict(v) for v in value]  # Recurse on lists
      attrs[field.name] = value
    return attrs


# Register mapping type -> asdict_fn
# SpecBase objects can be serialized to nested dict through `.asdict()`
# All types defined here will be automatically serialized
# * type: Class to serialize to dict
# * asdict_fn: Function which serialize the object to dict/value
_REGISTER: Dict[Type[_T], Callable[[_T], Any]] = {
    SpecBase: lambda x: x.asdict(),
    datetime.datetime: lambda x: x.isoformat(),  # pytype: disable=not-supported-yet
}


def _isregistered(obj: Any) -> bool:
  """Returns True if the object is registered (should be serialized)."""
  return isinstance(obj, tuple(_REGISTER.keys()))


def _asdict(obj: Any) -> Any:
  """Serialize the registered object to dict."""
  for cls, asdict_fn in _REGISTER.items():
    if isinstance(obj, cls):
      return asdict_fn(obj)
  raise TypeError(f'Unrecognized type: {obj}')
