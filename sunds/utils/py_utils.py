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

"""Python utils."""

import functools
import sys
from typing import Any, Callable

_Fn = Callable[..., Any]


def is_notebook() -> bool:
  """Returns True if running in a notebook (Colab, Jupyter) environment."""
  # Use sys.module as we do not want to trigger an import (slow)
  IPython = sys.modules.get('IPython')  # pylint: disable=invalid-name
  # Check whether we're not running in a IPython terminal
  if IPython:
    get_ipython_result = IPython.get_ipython()
    if get_ipython_result and 'IPKernelApp' in get_ipython_result.config:
      return True
  return False


# Do not use `TypeVar` as `map_fn` modify the signature. Is there a better
# typing annotation ?
def map_fn(fn: _Fn) -> _Fn:
  """Decorator around map_fn.

  Map function are function to apply inside `map` without the
  `functools.partial` boilerplate.

  Example:

  ```
  @sunds.utils.map_fn
  def add_prefix(val, *, prefix):
    return prefix + val

  iterable = ['a', 'b', 'c']
  assert map(add_prefix(prefix='_'), iterable) == ['_a', '_b', '_c']

  # The function can still be applied individually
  assert add_prefix('abc', prefix='_') == '_abc'
  ```

  Args:
    fn: Function to decorate

  Returns:
    The decorated function.
  """

  @functools.wraps(fn)
  def decorated(*args, **kwargs):
    if args:
      return fn(*args, **kwargs)
    else:
      return functools.partial(fn, **kwargs)

  return decorated
