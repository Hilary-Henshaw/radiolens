# Contributing to radiolens

Thank you for considering a contribution to radiolens. This project aims to
demonstrate rigorous, production-quality medical AI engineering, and
contributions are expected to uphold those standards. Please read this guide
carefully before opening a pull request.

---

## Code of Conduct

All contributors are expected to abide by the project's
[Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, constructive, and
assume good faith. The project maintainers reserve the right to remove
contributions or revoke access for conduct that violates the code of conduct.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/radiolens.git
cd radiolens
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install in Editable Mode with Dev Extras

```bash
pip install -e ".[dev]"
```

The `dev` extra installs ruff, mypy, pytest, pytest-cov, pre-commit, and all
optional dependency groups needed to run the full test suite locally.

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

Pre-commit runs ruff (lint + format), mypy, and a few file hygiene checks on
every commit. You will not be able to commit code that fails these checks
without explicitly bypassing the hooks (see the development guide for when
that is acceptable).

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, tagged releases only. Direct pushes are blocked. |
| `develop` | Integration branch. All feature and fix PRs target this branch. |
| `feature/<short-description>` | New features and enhancements. |
| `fix/<short-description>` | Bug fixes. |
| `docs/<short-description>` | Documentation-only changes. |
| `chore/<short-description>` | Tooling, CI, dependency updates. |

Work should always start from a fresh branch off `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature
```

---

## Commit Message Format

radiolens follows [Conventional Commits](https://www.conventionalcommits.org/).
Every commit message must have a type prefix. The subject line should be
written in imperative mood, lowercase, and must not end with a full stop.

```
<type>[optional scope]: <short description>

[optional body — wrap at 72 chars]

[optional footer — e.g. Closes #42]
```

### Allowed Types

| Type | When to Use |
|------|------------|
| `feat` | A new feature visible to users or callers |
| `fix` | A bug fix |
| `docs` | Documentation only — no source changes |
| `test` | Adding or refining tests, no production code change |
| `refactor` | Code restructuring with no behaviour change |
| `perf` | A change that improves performance |
| `chore` | Build tooling, CI, dependency bumps |
| `ci` | Changes to GitHub Actions workflows |

### Examples

```
feat(api): add streaming response option to /classify endpoint

fix(metrics): correct off-by-one in bootstrap resample indexing

docs: add troubleshooting entry for DICOM photometric errors

test(training): add slow integration test for full training loop

refactor(core): extract certainty tier logic into standalone function
```

---

## Testing Requirements

### Before Submitting a PR

Your branch must pass the fast test suite with no failures:

```bash
make test-fast
```

This runs only unit tests (no TensorFlow model loading, no disk I/O) and
completes in under 30 seconds. Failing the fast suite is a blocking review
criterion.

### Slow Tests

Integration tests that load a real model or touch the filesystem are marked
`@pytest.mark.slow`. They are not required to pass locally (many contributors
will not have weights available), but the CI pipeline runs them on every PR
against `develop`. If your change affects training, inference, or the API
server, please also run:

```bash
make test-slow   # requires RADIOLENS_MODEL_WEIGHTS_PATH to be set
```

### Coverage

The coverage gate is 80% for the `src/radiolens` package. New code you add
must be covered by tests. If a code path is genuinely untestable (e.g. an OS
signal handler), mark it with `# pragma: no cover` and add a comment
explaining why.

---

## Code Style

### Formatter and Linter

radiolens uses [ruff](https://docs.astral.sh/ruff/) for both formatting and
linting. Configuration lives in `pyproject.toml`. The pre-commit hook runs
ruff automatically; you can also run it manually:

```bash
ruff check src/ tests/           # lint
ruff format src/ tests/          # format
```

### Line Length

Maximum **79 characters** for Python source files. This is enforced by ruff.
Markdown documentation may use any line length.

### Type Hints

Every function and method signature must carry complete type annotations.
Return types are required even for functions that return `None`. Use
`from __future__ import annotations` at the top of each module to enable
forward references without runtime overhead.

```python
from __future__ import annotations

def compute_threshold(
    probabilities: list[float],
    target_sensitivity: float = 0.95,
) -> float:
    ...
```

### Logging

Use `structlog` for all logging. Never use `print()` for diagnostic output in
library code. Logger instances must be module-level:

```python
import structlog

log = structlog.get_logger(__name__)

def my_function() -> None:
    log.info("starting inference", image_shape=(224, 224, 3))
    log.debug("intermediate value", score=0.873)
    log.warning("low confidence prediction", certainty_tier="LOW")
```

Key naming convention: snake_case, descriptive, no abbreviations. Avoid
generic keys like `msg` or `data`.

---

## Pull Request Checklist

Before marking your PR as ready for review, confirm each item:

- [ ] Branch is based on `develop`, not `main`
- [ ] Commit messages follow Conventional Commits format
- [ ] `make test-fast` passes with no failures or warnings
- [ ] New code has tests; coverage has not decreased
- [ ] Type annotations are present on all new functions and methods
- [ ] `ruff check` and `ruff format` produce no output (pre-commit is clean)
- [ ] `mypy src/` produces no new errors
- [ ] Docstrings are present on all public classes and functions
- [ ] If the change affects the REST API, `docs/api-reference.md` is updated
- [ ] If the change adds a new env var, `docs/configuration.md` is updated
- [ ] If the change is user-visible, an entry has been added to `CHANGELOG.md`
  under `[Unreleased]`
- [ ] PR description explains *why* the change is needed, not just what it does

---

## How to Add a New Metric to ClinicalMetricsReport

1. Open `src/radiolens/evaluation/metrics.py`.

2. Add the computation to `compute_binary_metrics()`. The function receives
   `y_true: np.ndarray` and `y_pred_proba: np.ndarray` and returns a
   `ClinicalMetricsReport`. Add your metric calculation inside this function.

   ```python
   from sklearn.metrics import balanced_accuracy_score

   balanced_acc = float(
       balanced_accuracy_score(y_true, y_pred_binary)
   )
   ```

3. Add the new field to `ClinicalMetricsReport`. It is a `dataclass` (or
   Pydantic model); add the field with its type and a docstring:

   ```python
   balanced_accuracy: float
   """Balanced accuracy, correcting for class imbalance."""
   ```

4. Pass the computed value when constructing the return object.

5. If the metric should appear in the API `/performance` response, add it to
   the `PerformanceStats` contract in `src/radiolens/api/contracts.py`.

6. Add a unit test in `tests/unit/test_metrics.py` covering at least:
   - A perfect-prediction case
   - A worst-case (inverted labels) case
   - A realistic case with known expected value

7. Update `docs/api-reference.md` if the field appears in any response schema.

---

## How to Add a New Visualization to DiagnosticVisualizer

1. Open `src/radiolens/evaluation/visualizer.py`.

2. Add a method to `DiagnosticVisualizer` following the existing pattern.
   Every plot method must:
   - Accept `output_path: Path` as its only required argument
   - Return `None`
   - Use `matplotlib` with the project's shared style context
   - Save the figure with `fig.savefig(output_path, dpi=150, bbox_inches="tight")`
   - Call `plt.close(fig)` before returning to avoid memory leaks

   ```python
   def plot_calibration_curve(
       self,
       output_path: Path,
   ) -> None:
       """Plot reliability diagram for probability calibration."""
       fig, ax = plt.subplots(figsize=(6, 6))
       # ... plotting logic ...
       fig.savefig(output_path, dpi=150, bbox_inches="tight")
       plt.close(fig)
   ```

3. If the plot should be part of the standard `save_all_plots()` call, add
   it to the list inside that method.

4. Add a unit test in `tests/unit/test_visualizer.py` that:
   - Constructs a `DiagnosticVisualizer` with synthetic data
   - Calls the new method with a `tmp_path` output
   - Asserts the output file exists and is non-empty

5. Update `docs/development.md` if the plot has non-obvious parameters or
   configuration.

---

## Reporting Issues

### Bug Reports

Open a GitHub issue using the **Bug Report** template. Include:

- A minimal reproducible example (MRE) — the smallest code snippet that
  demonstrates the problem
- The full traceback if an exception is raised
- Your Python version (`python --version`)
- Your radiolens version (`pip show radiolens`)
- Your operating system and TensorFlow version

### Feature Requests

Open a GitHub issue using the **Feature Request** template. Describe:

- The problem you are trying to solve (not the solution you have in mind)
- Who would benefit from this feature
- Any constraints or edge cases worth considering

### Security Vulnerabilities

Do not open a public issue for security vulnerabilities. Email
`conduct@radiolens.dev` with a description of the issue and we will respond
within 72 hours.
