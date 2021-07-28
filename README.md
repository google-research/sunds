# Scene understanding datasets

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

*This is not an official Google product.*
