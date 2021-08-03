# Scene understanding datasets

[![Unittests](https://github.com/google-research/sunds/actions/workflows/pytest.yml/badge.svg)](https://github.com/google-research/sunds/actions/workflows/pytest.yml)
[![PyPI version](https://badge.fury.io/py/tensorflow-datasets.svg)](https://badge.fury.io/py/tensorflow-datasets)

SunDs is a collection of ready-to-use datasets for scene understanding tasks (3d
object detection, semantic segmentation, nerf rendering,...). It provides:

*   An API to easily load datasets to feed into your ML models.
*   A collection of ready-to-use datasets.
*   Helper tools to create new datasets.

```python
import sunds

ds = sunds.load('nerf_synthetic/lego', split='train', task=sunds.task.Nerf())
for ex in ds:
  ex['ray_origin']
```

To use sunds, see the documentation:

*  [Intro guide](https://github.com/google-research/sunds/blob/master/docs/intro.md)
*  [Nerf guide](https://github.com/google-research/sunds/blob/master/docs/nerf.md)

*This is not an official Google product.*
