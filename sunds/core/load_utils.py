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

"""Load API.

For familiarity, those utils try to match the TFDS load API.
"""

import functools
from typing import Optional

from sunds import utils
from sunds.core import dataset_builder
from sunds.core import tasks
from sunds.typing import PathLike, Split, Tree  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


def load(
    name: str,
    *,
    split: Tree[Split],
    data_dir: Optional[PathLike] = None,
    task: Optional[tasks.Task] = None,
    # TODO(epot): Add arguments to match TFDS signature
) -> tf.data.Dataset:
  """Load a sunds `tf.data.Dataset`.

  Args:
    name: Dataset name
    split: Split to load
    data_dir: Root TFDS directory
    task: Task definition. Control the subset of spec read from the dataset,
      the pre-processing to apply,... If none, read the frames dataset.

  Returns:
    ds: The `tf.data.Dataset` pipeline.
  """
  builder_ = builder(
      name=name,
      data_dir=data_dir,
  )
  return builder_.as_dataset(split=split, task=task)


def builder(
    name: str,
    *,
    use_code: bool = False,
    **builder_kwargs,
) -> dataset_builder.DatasetBuilder:
  """Get the dataset builder.

  This function assume that the `data_dir/` contain 2 datasets `<name>_scenes`
  and `<name>_frames`.

  Args:
    name: Dataset name, optionally with config, version
    use_code: If `True`, load the datasets from the source code (required to
      generate the dataset). Otherwise, load the dataset from pre-generated
      files on disk (last version loaded, unless specified otherwise).
    **builder_kwargs: Additional kwargs for `tfds.core.DatasetBuilder`

  Returns:
    builder: The instantiated dataset builder.
  """
  # Load frame and scenes dataset
  builder_name, builder_kwargs = tfds.core.naming.parse_builder_name_kwargs(
      name, **builder_kwargs
  )
  if use_code:
    # Code requested: Dynamically import the datasets
    # Note: Should be careful on colab with adhoc imports. Dataset should be
    # registered inside the adhoc scope.
    base_module_name = _get_base_code_module(builder_name)
    scene_module = f'{base_module_name}.scenes_builder'
    frame_module = f'{base_module_name}.frames_builder'
    scene_cls = tfds.core.community.builder_cls_from_module(scene_module)
    frame_cls = tfds.core.community.builder_cls_from_module(frame_module)
    scene_builder = scene_cls(**builder_kwargs)  # pytype: disable=not-instantiable
    frame_builder = frame_cls(**builder_kwargs)  # pytype: disable=not-instantiable
  else:
    # Otherwise, datasets not registered. `tfds.builder` will restore the
    # last dataset found in `data_dir/`
    scene_builder = functools.partial(
        tfds.builder,
        f'{builder_name}_scenes',
        **builder_kwargs,
    )
    frame_builder = functools.partial(
        tfds.builder,
        f'{builder_name}_frames',
        **builder_kwargs,
    )

  return dataset_builder.DatasetBuilder(
      scene_builder=dataset_builder.LazyBuilder(scene_builder),
      frame_builder=dataset_builder.LazyBuilder(frame_builder),
  )


def _get_base_code_module(builder_name: tfds.core.naming.DatasetName) -> str:
  """Find the code location of the requested dataset."""
  builder_name = str(builder_name)
  module_path = utils.sunds_dir() / 'datasets'
  if (module_path / builder_name).exists():
    return f'sunds.datasets.{builder_name}'
  else:
    raise ValueError(f'Could not find dataset code for {builder_name}')
