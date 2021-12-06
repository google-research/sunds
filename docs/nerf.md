# Nerf

## Usage

`sunds.tasks.Nerf` contain the logic to extract/compute the features required
for nerf-based model, including:

*   Auto-compute the rays origins/directions if not present in the dataset.
*   Normalize the rays acording to the scene bounding boxes

By default, `sunds.tasks.Nerf` returns the following fields:

```python
ds = sunds.load(..., task=sunds.tasks.Nerf())
ds.element_specs == {
    # Ray origin/direction
    'ray_origins': tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
    'ray_directions': tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
    # RGB values
    'color_image': tf.TensorSpec(shape=(h, w, 3), dtype=tf.uint8),
    # Additional metadata to identify the image
    'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
    'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
    'camera_name': tf.TensorSpec(shape=(), dtype=tf.string),
}
```

## Arguments

### `yield_mode`

Some scene understanding datasets might contain multiple camera views per frame.
For example, Waymo data can have `front_camera`, `back_camera`,... for each
examples.

By default, each camera is yield individually with an additional `camera_name`
field. It is possible to change this behavior:

```python
ds = sunds.load(..., task=sunds.tasks.Nerf(yield_mode=...))
ds.element_specs.shape == {
    'ray_origins': (*batch_shape, 3),
    'ray_directions': (*batch_shape, 3),
    'color_image': (*batch_shape, 3),
    'scene_name': tf.string,
    'frame_name': tf.string,
    'camera_name': tf.string,
}

ds = ds.shuffle(ray_buffer_size)  # Shuffle rays across images
```

Depending on `yield_mode`, `batch_shape` can be:

* `YieldMode.RAY` (or `'ray'`): `()`
* `YieldMode.IMAGE` (or `'image'`): `(h, w)`
* `YieldMode.STACKED` (or `'dict'`): `(num_cameras, h, w)`

`yield_mode` can also be  `YieldMode.DICT` (or `'dict'`) in which case, the original dict is returned.

```python
ds = sunds.load(..., task=sunds.tasks.Nerf(yield_mode='dict'))
ds.element_specs.shape == {
    'cameras': {
        'front_camera': {
            'ray_origins': (h, w, 3),
            'ray_directions': (h, w, 3),
            'color_image': (h, w, 3),
        },
        'back_camera': {
            'ray_origins': (h, w, 3),
            'ray_directions': (h, w, 3),
            ...
        },
        ...
    },
    'scene_name': (),
    'frame_name': (),
    'camera_name': (),
}
```

### `normalize_rays`

By default, the rays are returned as-is (without any transformation). Setting
`normalize_rays=True` will apply:

*   Scene normalization: Normalizing the ray origins/directions such as all
    objects within the scenes are contained inside the `[-1, 1]` 3D box (Note
    that the cameras can still be outside this `[-1, 1]` box). This can be used
    by model to sample points only within this range.
*   Euclidian normalization: Applying `tf.norm` on the rays directions.

For more complex normalization, `normalize_rays` can also be a custom object. Please contact us (e.g. open a GitHub issue) if you need specific features.

### `additional_camera_specs` / `additional_frame_specs`

Scene understanding datasets contain many fields, but only a small subset are
required for a specific task. By default, `sunds.tasks.Nerf` only return the
minimal subset needed to train a model but you might want to use additional
fields.

This can be done through the `additional_frame_specs` kwarg:

```python
# Specify the additional fields to return
task = sunds.tasks.Nerf(
    additional_camera_specs={'category_image', 'depth'},
    additional_frame_specs={'timestamp'},
)

ds = sunds.load(..., task=task)
ds.element_specs == {
    # Default field returned
    'ray_origins': tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
    'ray_directions': tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
    'color_image': tf.TensorSpec(shape=(h, w, 3), dtype=tf.uint8),
    'scene_name': tf.TensorSpec(shape=(), dtype=tf.string),
    'frame_name': tf.TensorSpec(shape=(), dtype=tf.string),
    'camera_name': tf.TensorSpec(shape=(), dtype=tf.string),
    # Additional camera field returned
    'ray_directions': tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
    'color_image': tf.TensorSpec(shape=(h, w, 3), dtype=tf.uint8),
    # Additional frame field returned
    'timestamp': tf.TensorSpec(shape=(), dtype=tf.float32),
}
```

Nested feature selection is also supported. Accepted structure are the same as
[`tfds.decode.PartialDecoding`](https://www.tensorflow.org/datasets/decode#only_decode_a_sub-set_of_the_features).
