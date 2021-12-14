# Copyright 2021 The sunds Authors.
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

"""SpecDict util."""

from typing import Any, Callable, Union

from sunds import utils
from sunds.typing import FeatureSpecs, FeaturesOrBool, LabelArg  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds

_Factory = Callable[[Any], FeatureSpecs]


class SpecDict(dict):
  """Spec dict util is like a dict with util methods to manipulate specs."""

  def maybe_set(
      self,
      key: str,
      value: Union[FeaturesOrBool, Any],
      default: Union[FeaturesOrBool, _Factory] = False,
  ) -> None:
    """Updates the dict depending on the content of `value`.

    This function update the `dict` depending on the content of `value`. If
    `value` is:

    * `False`: Dict not updated
    * `True`: Dict updated with `default`
    * `FeatureConnector`: Dict updated with `value`

    Default can also be a factory. This allow lazy-initializing the feature
    using additional info provided by the user.
    In which case, if `value` is not any of the above types, Dict is updated
    with `default(value)`

    Example:

    ```python
    specs = SpecDict()
    specs.maybe_set(  # value is False  ->  Dict not updated
        'a',
        False,
        tfds.features.Image(),
    )
    specs.maybe_set(  # value is Feature  ->  Dict updated with value.
        'b',
        tfds.features.Image(shape=(h, w, 3), encoding_format='png'),
        tfds.features.Image(),
    )
    # default is factory  ->  Dict updated with LabeledImage(labels=value).
    specs.maybe_set(
        'c',
        ['background', 'car', 'street'],
        lambda labels: tfds.features.LabeledImage(labels=labels),
    )

    assert specs == {
        'b': tfds.features.Image(shape=(h, w, 3), encoding_format='png'),
        'c': tfds.features.LabeledImage(labels=['background', 'car', 'street']),
    }
    ```

    Args:
      key: Dict key to set
      value: FeatureConnector with which update the dict, or bool.
      default: If `value` is `bool` and `True`, then the dict is updated with
        `default`.
    """
    if _is_features(value):
      # Feature connector passed
      self[key] = value
    elif isinstance(value, bool):
      if not value:  # Dict not updated
        return
      if not default:
        raise ValueError(f'{key}=True, but no default provided. '
                         f'{key} should be feature instead.')
      if callable(default):  # Default is a factory
        with tfds.core.utils.try_reraise(f'Error building `{key}`: '):
          default = default(None)
      # Update the dict with the default value
      if not _is_features(default):
        raise TypeError(
            f'Invalid default for `{key}`: {default}. Expected nested '
            'features.')
      self[key] = default
    elif callable(default):
      # Value and factory given, forward the value to the factory.
      with tfds.core.utils.try_reraise(f'Error building `{key}`: '):
        self[key] = default(value)
    else:
      raise TypeError(
          f"Invalid key '{key}': {value}\nExpected bool or feature.")

  def maybe_update(
      self,
      other: FeaturesOrBool,
      default: FeaturesOrBool = False,
  ) -> None:
    """Update the `dict` depending on the content of `other`.

    If `other` is:

    * `False`: Do not update the `dict`
    * `True`: Update the dict with `default`
    * `Dict[str, FeatureConnector]`: Dict updated with `other`

    Args:
      other: FeatureConnectors with which update the dict, or bool.
      default: If `other` is `bool` and `True`, then the dict is updated with
        `default`.
    """
    if isinstance(other, bool):
      if not other:
        return
      if default is False:  # pylint: disable=g-bool-id-comparison
        raise ValueError(
            'Update set to True, but no default provided. Should provide '
            'feature instead.')
      if not _is_features(default):
        raise TypeError(
            f'Invalid default for {default}. Expected nested features.')
      self.update(default)
    else:
      self.update(other)


@utils.map_fn
def labeled_image(
    labels: LabelArg,
    **kwargs: Any,
) -> tfds.features.LabeledImage:
  return tfds.features.LabeledImage(labels=labels, **kwargs)


def _is_features(features: FeatureSpecs) -> bool:
  """Recursivelly check that the input is a nested feature."""
  features_cls = (tf.dtypes.DType, tfds.features.FeatureConnector)
  is_features_nested = tf.nest.map_structure(
      lambda f: isinstance(f, features_cls),
      features,
  )
  return all(tf.nest.flatten(is_features_nested))
