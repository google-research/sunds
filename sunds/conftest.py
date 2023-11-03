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

"""Pytest plugins globally available.

`conftest.py` file is automatically detected by Pytest and register
plugins (hooks and fixtures) common to all tests.
So all tests can use the setup/teardown functions defined here.

See: https://docs.pytest.org/en/latest/writing_plugins.html

"""

from typing import Iterator

from unittest import mock
import pytest
import sunds


@pytest.fixture(scope='session')
def lego_builder(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[sunds.core.DatasetBuilder]:
  """Dummy nerf synthetic dataset pre-generated."""
  data_path = tmp_path_factory.mktemp('global_nerf_synthetic_data_dir')

  dummy_path = sunds.utils.sunds_dir() / 'datasets/nerf_synthetic/dummy_data'

  builder = sunds.builder(
      'nerf_synthetic/lego',
      data_dir=data_path,
      use_code=True,
  )

  # Generate the dataset using the fake data
  with mock.patch(
      'tensorflow_datasets.download.DownloadManager.download_and_extract',
      return_value=dummy_path,
  ):
    builder.download_and_prepare()

  yield builder


@pytest.fixture(scope='session')
def lego_builder_frame_only(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[sunds.core.DatasetBuilder]:
  """Dummy nerf synthetic dataset pre-generated."""
  # Could instead reuse `lego_builder` and copy the frame dir.

  data_path = tmp_path_factory.mktemp('global_nerf_frame_only_data_dir')

  dummy_path = sunds.utils.sunds_dir() / 'datasets/nerf_synthetic/dummy_data'

  builder = sunds.builder(
      'nerf_synthetic/lego',
      data_dir=data_path,
      use_code=True,
  )

  # Generate the dataset using the fake data
  with mock.patch(
      'tensorflow_datasets.download.DownloadManager.download_and_extract',
      return_value=dummy_path,
  ):
    builder._frame_builder.download_and_prepare()  # pylint: disable=protected-access

  yield builder
