# Getting started

## Overview

Sunds is a collection of datasets for scene understanding tasks. Each dataset
contain information about one or multiple scenes. This can includes rgb images,
lidar points cloud, depth map, semantic maps,...

When loading a dataset, the user select the required task which will
automatically select the subsbset of features required for the task.

![overview](https://docs.google.com/drawings/d/1pSBIzbU2ZGRYq-w3wIlYsMgNOJrtEWrpXqyPkUJFpVs/export/png)

## Load a dataset

SunDs API is similar to [TFDS](https://www.tensorflow.org/datasets/overview)
with an additional `task` argument (see next section).

```python
ds = sunds.load('nerf_synthetic/lego', split='train', task=sunds.tasks.Nerf())

for ex in ds:
  ex['ray_origin']
```

Similarly to TFDS, sunds also has a lower level API. The above `sunds.load`
statement is equivalent to:

```python
builder = sunds.builder('nerf_synthetic/lego')
ds = builder.as_dataset(split='train', task=sunds.tasks.Nerf())
```

Like any `tf.data.Dataset`, you can apply additional transformation:

```python
ds = sunds.load('nerf_synthetic/lego', split='train', task=sunds.tasks.Nerf())
ds = ds.batch(10)
ds = ds.shuffle(buffer_size)
ds = ds.prefetch(1)

for ex in tfds.as_numpy(ds):
  ...
```

## Dataset specifications

All scene understanding datasets are composed of 2 sub-datasets:

*   The scenes dataset: containing the high level scene metadata (e.g. scene
    boundaries, mesh of the full scene,...)
*   The frames dataset: containing the individual examples within a scene (rgb
    image, bounding boxes,...)

## Tasks

### Concept

SunDs datasets can have many features (lidar, depth map, ...). However only a
small subset are require for a specific task (object detection, nerf,...).
Additionally, each task requires different pipeline pre-processing (e.g.
dynamically generate rays for nerf).

SunDs introduce the concept of **task** to inject the required transformation
logic. A task controls:

*   Which features of the dataset to use/decode (the other features are ignored)
*   Which transformation to apply to the pipeline

For example `sunds.task.Nerf()` will use the camera information of the dataset
to dynamically generate the rays origin/position and will returns a
`tf.data.Dataset` containing the `{'ray_origins': ..., 'ray_directions': ...,
}`.

```python
ds = sunds.load(
    'nerf_synthetic/lego',
    split='train',
    task=sunds.tasks.Nerf(yield_mode='ray'),  # Mode can be ray, image,...
)
assert ds.element_spec == {
    'ray_directions': tf.TensorSpec(shape=(3,), dtype=tf.float32),
    'ray_origins': tf.TensorSpec(shape=(3,), dtype=tf.float32),
    'color_image': tf.TensorSpec(shape=(3,), dtype=tf.uint8),
}
```

Tasks can be further customized during construction. In the above example,
`keep_img_dim=False` control whether to return examples batched at the pixel or
the image level.

### Using existing tasks

The easiest way to get started training a model is to use one of the existing
`sunds.task`. Look at the documentation for specific task:

*   [sunds.tasks.Nerf](nerf.md): To extract ray origins/positions as well as
    applying various normalizations.

### Creating a task

There are 2 ways of creating a Task.

*   By inheriting of `sunds.core.Task`: Lower level, but more flexible
*   By inheriting of `sunds.core.FrameTask`: Simpler

#### `sunds.core.Task`

`sunds.core.Task` is the abstract base class from which all task inherit. It
only has a single abstract method which allows to fully overwrite the
`as_dataset` method. So each task has full control over which dataset is
returned.

```python
class Task(abc.Abc):
  # The task has access to the scene and frame builders through attributes.
  scene_builder: tfds.core.DatasetBuilder
  frame_builder: tfds.core.DatasetBuilder

  @abc.abstractmethod
  def as_dataset(**kwargs) -> tf.data.Dataset:
    raise NotImplementedError
```

Example: a dummy task which load the frame dataset and apply batching.

```python
@dataclasses.dataclass
class DummyTask(sunds.core.Task):
  batch_size: int

  def as_dataset(self, **kwargs):
    # Forward the `split='train'`,... kwargs from `sunds.load`
    ds = self.frame_builder.as_dataset(**kwargs)
    ds = ds.batch(self.batch_size)
    return ds


ds = sunds.load(
    'nerf_synthetic/lego',
    split='train',
    task=DummyTask(batch_size=123),
)  # ds is now batched
```

#### `sunds.core.FrameTask`

In practice, most tasks can inherit from `sunds.task.FrameTask`. The task then
need to implement 2 methods:

*   `frame_specs`: Returns subset of the tfds.features to decode (all other
    features will be ignored)
*   `pipeline`: Eventual `tf.data` transformation to apply to the pipeline (e.g.
    compute rays)

`FrameTask` is more or less equivalent to:

```python
class FrameTask(Task):

  def as_dataset(**kwargs) -> tf.data.Dataset:
    ds = frame_builder.as_dataset(
        decoders=tfds.features.PartialDecoding(self.frame_specs),
        **kwargs,
    )
    ds = self.pipeline(ds)
    return ds
```

Example:

```python
class DummyFrameTask(sunds.core.FrameTask):
  """Dummy frame task which only load the scene name."""

  @property
  def frame_specs(self):
    return {
        'scene_name': tfds.features.Tensor(shape=(), dtype=tf.string),
    }

  def pipeline(self, ds):
    ds = ds.batch(3)
    return ds


ds = sunds.load(
    'nerf_synthetic/lego',
    split='train',
    task=DummyFrameTask(),
)
# Only the scene_name is returned.
assert ds.element_spec == {
    'scene_name': tf.TensorSpec(shape=(3,), tf.string),
}
```
