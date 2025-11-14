# Configuration Reference

radiolens is configured exclusively through environment variables with the
`RADIOLENS_` prefix, managed by a Pydantic `Settings` class in
`src/radiolens/config.py`. All values can also be placed in a `.env` file in
the working directory.

---

## Getting Started

Copy the provided example and edit as needed:

```bash
cp .env.example .env
```

The minimum required variable for inference is `RADIOLENS_MODEL_WEIGHTS_PATH`.

---

## Variable Reference

### Image Preprocessing

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_IMAGE_HEIGHT` | `224` | Input image height in pixels |
| `RADIOLENS_IMAGE_WIDTH` | `224` | Input image width in pixels |
| `RADIOLENS_IMAGE_CHANNELS` | `3` | Number of colour channels (RGB = 3) |

### Model Backbone

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_BACKBONE_NAME` | `"MobileNetV2"` | CNN backbone identifier (informational) |

### Classification Head

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_DENSE_LAYER_UNITS` | `128` | Number of units in the Dense layer |
| `RADIOLENS_FIRST_DROPOUT_RATE` | `0.3` | Dropout after GlobalAveragePooling2D |
| `RADIOLENS_SECOND_DROPOUT_RATE` | `0.2` | Dropout before the sigmoid output |

### Training

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_BATCH_SIZE` | `32` | Mini-batch size during training |
| `RADIOLENS_MAX_EPOCHS` | `25` | Maximum number of training epochs |
| `RADIOLENS_INITIAL_LEARNING_RATE` | `0.001` | Starting learning rate for Adam |
| `RADIOLENS_MINIMUM_LEARNING_RATE` | `1e-7` | Floor for `ReduceLROnPlateau` |
| `RADIOLENS_EARLY_STOP_PATIENCE` | `7` | Epochs without improvement before stopping |
| `RADIOLENS_LR_REDUCE_PATIENCE` | `4` | Epochs before reducing learning rate |
| `RADIOLENS_LR_REDUCE_FACTOR` | `0.5` | Multiplicative factor for LR reduction |
| `RADIOLENS_RANDOM_SEED` | `42` | Global random seed for reproducibility |

### Data Augmentation

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_AUG_ROTATION_DEGREES` | `20.0` | Max rotation in degrees |
| `RADIOLENS_AUG_WIDTH_SHIFT` | `0.2` | Fraction of width for horizontal shift |
| `RADIOLENS_AUG_HEIGHT_SHIFT` | `0.2` | Fraction of height for vertical shift |
| `RADIOLENS_AUG_ZOOM_RANGE` | `0.2` | Zoom factor range |
| `RADIOLENS_AUG_HORIZONTAL_FLIP` | `true` | Enable random horizontal flip |
| `RADIOLENS_AUG_BRIGHTNESS_LOWER` | `0.8` | Lower bound for brightness adjustment |
| `RADIOLENS_AUG_BRIGHTNESS_UPPER` | `1.2` | Upper bound for brightness adjustment |

### Dataset Splitting

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_TRAIN_FRACTION` | `0.7` | Proportion of data for training |
| `RADIOLENS_VALIDATION_FRACTION` | `0.2` | Proportion of data for validation |
| `RADIOLENS_TEST_FRACTION` | `0.1` | Proportion of data for testing |
| `RADIOLENS_ACCEPTED_IMAGE_SUFFIXES` | `[".jpeg",".jpg",".png"]` | Accepted file extensions |

> The three fractions must sum to exactly 1.0 — the `Settings` validator
> enforces this at startup.

### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_API_HOST` | `"0.0.0.0"` | Bind address for uvicorn |
| `RADIOLENS_API_PORT` | `8000` | Listen port (also read from `PORT`) |
| `RADIOLENS_MAX_UPLOAD_BYTES` | `10485760` (10 MB) | Maximum accepted file size |
| `RADIOLENS_MODEL_WEIGHTS_PATH` | `"model/best_weights.keras"` | Path to `.keras` or `.h5` weights |
| `RADIOLENS_CORS_ALLOW_ORIGINS` | `["*"]` | CORS allowed origins list |

> `RADIOLENS_API_PORT` also accepts the plain `PORT` variable, which is the
> convention used by Render and other PaaS platforms.

### Bootstrap Statistics

| Variable | Default | Description |
|----------|---------|-------------|
| `RADIOLENS_BOOTSTRAP_RESAMPLES` | `1000` | Number of bootstrap iterations |
| `RADIOLENS_BOOTSTRAP_CI_LEVEL` | `0.95` | Confidence interval coverage |

---

## Derived Properties

Two read-only properties are computed from the image dimension settings:

```python
settings.input_shape   # (224, 224, 3)
settings.target_size   # (224, 224)
```

These are used internally by `ImageNormalizer` and `ThoraxClassifier`.

---

## Accessing Settings in Code

Use `get_settings()` to obtain the module-level singleton:

```python
from radiolens.config import get_settings

settings = get_settings()
print(settings.model_weights_path)  # PosixPath('model/best_weights.keras')
print(settings.input_shape)         # (224, 224, 3)
```

For tests or tools that need isolated settings, instantiate `Settings`
directly:

```python
from radiolens.config import Settings

settings = Settings(
    model_weights_path="tests/fixtures/stub_weights.keras",
    max_epochs=1,
)
```
