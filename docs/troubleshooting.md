# Troubleshooting

---

## Model / Weights Issues

### `FileNotFoundError: Model weights not found at: model/best_weights.keras`

The model weights file is not included in the repository. You need to
provide it at the path indicated by `RADIOLENS_MODEL_WEIGHTS_PATH`.

**Options:**

```bash
# Download from Hugging Face Hub
pip install huggingface-hub
huggingface-cli download \
    ayushirathour/chest-xray-pneumonia-detection \
    best_chest_xray_model.h5 \
    --local-dir model/

# Then point the variable at the downloaded file
export RADIOLENS_MODEL_WEIGHTS_PATH=model/best_chest_xray_model.h5
```

Or train your own weights using the training scripts — see
[training-walkthrough](../examples/training-walkthrough/).

---

### `RuntimeError: Model is not initialised. Call build() or load_weights().`

You called `run_inference()` before loading weights.

```python
# Correct usage
classifier = ThoraxClassifier(settings)
classifier.load_weights(Path("model/best_weights.keras"))  # ← required
result = classifier.run_inference(pixel_array)
```

---

### `ValueError: Expected shape (224, 224, 3), got (256, 256, 3)`

The pixel array passed to `run_inference()` does not match the model's
expected input shape. Always use `ImageNormalizer` to preprocess images — it
handles resizing automatically.

```python
normalizer = ImageNormalizer(settings)
pixel_array = normalizer.from_path(Path("xray.jpg"))
result = classifier.run_inference(pixel_array)
```

---

## API Issues

### `400 Unsupported file type`

The uploaded file has an extension that is not in
`{".jpeg", ".jpg", ".png", ".dcm"}`. Rename the file or convert it to a
supported format.

---

### `422 Cannot decode image`

The file has an accepted extension but cannot be opened as an image. Common
causes:

- Truncated or corrupted file
- A DICOM file with an unsupported transfer syntax
- A text file accidentally given an `.jpg` extension

---

### API returns `"status": "degraded"` on `/health`

The model was not loaded during startup. Check:

1. `RADIOLENS_MODEL_WEIGHTS_PATH` points to an existing file.
2. The file is readable by the process.
3. Check startup logs for `FileNotFoundError` or `RuntimeError`.

---

### `CORS` errors in the browser

The default `RADIOLENS_CORS_ALLOW_ORIGINS=["*"]` allows all origins. If
you have restricted it, add your front-end origin:

```bash
RADIOLENS_CORS_ALLOW_ORIGINS='["https://your-app.example.com"]'
```

---

## TensorFlow / Keras Issues

### `ImportError: TensorFlow / Keras is required for radiolens`

Install the `api` extras:

```bash
pip install "radiolens[api]"
```

Or install TensorFlow directly:

```bash
pip install tensorflow>=2.13.0,<2.16
```

---

### Slow inference on CPU

MobileNetV2 is designed for efficiency, but CPU inference on large batches
is slower than GPU. For production throughput:

- Use `--workers 2` (or more) with uvicorn to handle concurrent requests.
- Consider TensorFlow Lite conversion (on the roadmap) for edge deployment.

---

## Data / Training Issues

### `FileNotFoundError: No valid class folders found under: ...`

`RadiographBalancer` expects subdirectories named exactly `normal`,
`NORMAL`, `pneumonia`, or `PNEUMONIA`. Check your dataset structure:

```
data/raw/
├── NORMAL/
│   ├── img001.jpg
│   └── ...
└── PNEUMONIA/
    ├── img002.jpg
    └── ...
```

---

### Split fractions validation error at startup

```
ValueError: train_fraction + validation_fraction + test_fraction must equal 1.0
```

The three fractions must sum exactly to 1.0. With floating-point defaults
this is checked with a tolerance of 1e-6. Verify your `.env`:

```bash
RADIOLENS_TRAIN_FRACTION=0.7
RADIOLENS_VALIDATION_FRACTION=0.2
RADIOLENS_TEST_FRACTION=0.1
```

---

## Test Issues

### Tests fail with `ImportError` for `radiolens`

The package is not installed in the active environment. Run:

```bash
pip install -e ".[dev]"
```

---

### Coverage below 80%

The CI gate fails if line coverage drops below 80%. Add tests for any new
code paths before opening a pull request. Run locally with:

```bash
pytest --cov=radiolens --cov-report=term-missing
```
