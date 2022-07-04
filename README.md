# Scene understanding datasets

[![Unittests](https://github.com/google-research/sunds/actions/workflows/pytest.yml/badge.svg)](https://github.com/google-research/sunds/actions/workflows/pytest.yml)
[![PyPI version](https://badge.fury.io/py/sunds.svg)](https://badge.fury.io/py/sunds)

SunDs is a collection of ready-to-use datasets for scene understanding tasks (3d
object detection, semantic segmentation, nerf rendering,...). It provides:

*   An API to easily load datasets to feed into your ML models.
*   A collection of ready-to-use datasets.
*   Helper tools to create new datasets.

```python
import sunds

ds = sunds.load('kubric:nerf_synthetic/lego', split='train', task=sunds.tasks.Nerf())
for ex in ds:
  ex['ray_origin']
```

To use sunds, see the documentation:

*  [Intro guide](https://github.com/google-research/sunds/blob/master/docs/intro.md)
*  [Nerf guide](https://github.com/google-research/sunds/blob/master/docs/nerf.md)

## Load datasets

Some datasets are pre-processed and published directly in
`gs://kubric-public/tfds`. You can stream them directly from GCS with:

```python
sunds.load('kubric:nerf_synthetic/lego')
```

The `kubric:` prefix is just an alias for

```python
sunds.load('nerf_synthetic/lego', data_dir='gs://kubric-public/tfds')
```

For best performance, it's recommended to copy the data locally with
[gsutil](https://cloud.google.com/storage/docs/gsutil_install):

```sh
DATA_DIR=~/tensorflow_datasets/
mkdir $DATA_DIR
gsutil -m cp -r gs://kubric-public/tfds/nerf_synthetic/ $DATA_DIR
```

After the data has been copied locally, it can be loaded directly.

```python
sunds.load('nerf_synthetic/lego')
```

If you copy locally to another folder than `~/tensorflow_datasets/`,
you'll have to specify `data_dir='/path/to/tfds/'`.

*This is not an official Google product.*
