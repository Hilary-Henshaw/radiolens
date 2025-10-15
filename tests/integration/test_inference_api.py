"""Integration tests for the radiolens inference REST API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Keras/TF required at import of detector; skip module if absent.
pytest.importorskip(
    "keras",
    reason="TensorFlow/Keras not installed — skipping API tests",
)

from radiolens.api.server import create_api_app  # noqa: E402
from radiolens.core.detector import InferenceResult  # noqa: E402


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Return a MagicMock classifier that returns a fixed InferenceResult."""
    clf = MagicMock()
    clf.run_inference.return_value = InferenceResult(
        label="NORMAL",
        probability=0.12,
        confidence=0.88,
        certainty_tier="HIGH",
    )
    return clf


@pytest.fixture
def api_client(mock_classifier: MagicMock) -> TestClient:
    """TestClient with model loading bypassed via DI override."""
    from radiolens.api import providers
    from radiolens.api.providers import provide_classifier

    app = create_api_app()
    # Override classifier dependency
    app.dependency_overrides[provide_classifier] = lambda: mock_classifier
    # Patch ThoraxClassifier so the lifespan instantiates the mock instead
    # of trying to load weights from disk (which don't exist in CI).
    with (
        patch.object(
            providers, "ThoraxClassifier", return_value=mock_classifier
        ),
        TestClient(app, raise_server_exceptions=True) as client,
    ):
        yield client


# --------------------------------------------------------- /health


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(
        self,
        api_client: TestClient,
    ) -> None:
        """Health endpoint returns HTTP 200."""
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(
        self,
        api_client: TestClient,
    ) -> None:
        """Response body contains a 'status' field."""
        response = api_client.get("/api/v1/health")
        body = response.json()
        assert "status" in body
        assert body["status"] in {"healthy", "degraded"}

    def test_health_response_has_version_field(
        self,
        api_client: TestClient,
    ) -> None:
        """Response body contains a 'version' field."""
        response = api_client.get("/api/v1/health")
        body = response.json()
        assert "version" in body
        assert isinstance(body["version"], str)

    def test_health_model_loaded_is_bool(
        self,
        api_client: TestClient,
    ) -> None:
        """model_loaded field is a boolean."""
        response = api_client.get("/api/v1/health")
        body = response.json()
        assert isinstance(body["model_loaded"], bool)

    def test_health_uptime_seconds_is_numeric(
        self,
        api_client: TestClient,
    ) -> None:
        """uptime_seconds field is a non-negative number."""
        response = api_client.get("/api/v1/health")
        body = response.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0.0


# ---------------------------------------------------------- /info


class TestInfoEndpoint:
    """Tests for GET /api/v1/info."""

    def test_info_returns_200(
        self,
        api_client: TestClient,
    ) -> None:
        """Info endpoint returns HTTP 200."""
        response = api_client.get("/api/v1/info")
        assert response.status_code == 200

    def test_info_backbone_is_mobilenetv2(
        self,
        api_client: TestClient,
    ) -> None:
        """backbone field is 'MobileNetV2'."""
        response = api_client.get("/api/v1/info")
        body = response.json()
        assert body["backbone"] == "MobileNetV2"

    def test_info_output_classes_contains_normal_and_pneumonia(
        self,
        api_client: TestClient,
    ) -> None:
        """output_classes contains both 'NORMAL' and 'PNEUMONIA'."""
        response = api_client.get("/api/v1/info")
        body = response.json()
        classes = body["output_classes"]
        assert "NORMAL" in classes
        assert "PNEUMONIA" in classes

    def test_info_input_shape_has_three_elements(
        self,
        api_client: TestClient,
    ) -> None:
        """input_shape is a list of exactly three integers."""
        response = api_client.get("/api/v1/info")
        body = response.json()
        shape = body["input_shape"]
        assert isinstance(shape, list)
        assert len(shape) == 3

    def test_info_cross_operator_accuracy_in_valid_range(
        self,
        api_client: TestClient,
    ) -> None:
        """cross_operator_accuracy is a float in [0, 1]."""
        response = api_client.get("/api/v1/info")
        body = response.json()
        acc = body["cross_operator_accuracy"]
        assert 0.0 <= acc <= 1.0


# --------------------------------------------------- /performance


class TestPerformanceEndpoint:
    """Tests for GET /api/v1/performance."""

    def test_performance_returns_200(
        self,
        api_client: TestClient,
    ) -> None:
        """Performance endpoint returns HTTP 200."""
        response = api_client.get("/api/v1/performance")
        assert response.status_code == 200

    def test_performance_external_accuracy_is_expected_value(
        self,
        api_client: TestClient,
    ) -> None:
        """external_accuracy is the published value 0.860."""
        response = api_client.get("/api/v1/performance")
        body = response.json()
        assert body["external_accuracy"] == pytest.approx(0.860)

    def test_performance_external_sensitivity_is_expected_value(
        self,
        api_client: TestClient,
    ) -> None:
        """external_sensitivity is the published value 0.964."""
        response = api_client.get("/api/v1/performance")
        body = response.json()
        assert body["external_sensitivity"] == pytest.approx(0.964)

    def test_performance_bootstrap_p_value_is_expected(
        self,
        api_client: TestClient,
    ) -> None:
        """bootstrap_p_value is the published value 0.978."""
        response = api_client.get("/api/v1/performance")
        body = response.json()
        assert body["bootstrap_p_value"] == pytest.approx(0.978)

    def test_performance_has_n_external_samples(
        self,
        api_client: TestClient,
    ) -> None:
        """n_external_samples is a positive integer."""
        response = api_client.get("/api/v1/performance")
        body = response.json()
        assert body["n_external_samples"] > 0

    def test_performance_internal_accuracy_above_point_nine(
        self,
        api_client: TestClient,
    ) -> None:
        """Published internal accuracy is above 0.9."""
        response = api_client.get("/api/v1/performance")
        body = response.json()
        assert body["internal_accuracy"] > 0.9


# ---------------------------------------------------- /classify


class TestClassifyEndpoint:
    """Tests for POST /api/v1/classify."""

    def test_valid_jpeg_returns_200(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """Uploading a valid JPEG returns HTTP 200."""
        response = api_client.post(
            "/api/v1/classify",
            files={
                "file": (
                    "xray.jpg",
                    jpeg_bytes,
                    "image/jpeg",
                )
            },
        )
        assert response.status_code == 200

    def test_valid_png_returns_200(
        self,
        api_client: TestClient,
        tmp_path: Path,
    ) -> None:
        """Uploading a valid PNG returns HTTP 200."""
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "xray.png"
        Image.fromarray(arr, mode="RGB").save(img_path)
        png_bytes = img_path.read_bytes()

        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_response_contains_label_field(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """Response body contains a 'label' field."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert "label" in response.json()

    def test_response_label_is_normal_or_pneumonia(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """label field is either 'NORMAL' or 'PNEUMONIA'."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.json()["label"] in {"NORMAL", "PNEUMONIA"}

    def test_response_probability_in_zero_one_range(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """probability field is in [0.0, 1.0]."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        prob = response.json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_response_confidence_gte_half(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """confidence is max(p, 1-p) so it must be >= 0.5."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert response.json()["confidence"] >= 0.5

    def test_response_contains_clinical_note(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """clinical_note field is present and non-empty."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        note = response.json().get("clinical_note", "")
        assert len(note) > 0

    def test_response_contains_model_version(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """model_version field is present and non-empty."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        version = response.json().get("model_version", "")
        assert len(version) > 0

    def test_unsupported_file_type_returns_400(
        self,
        api_client: TestClient,
    ) -> None:
        """Uploading a .txt file returns HTTP 400."""
        response = api_client.post(
            "/api/v1/classify",
            files={
                "file": (
                    "report.txt",
                    b"some text",
                    "text/plain",
                )
            },
        )
        assert response.status_code == 400

    def test_invalid_image_data_returns_422(
        self,
        api_client: TestClient,
    ) -> None:
        """Sending garbage bytes labelled as JPEG returns HTTP 422."""
        response = api_client.post(
            "/api/v1/classify",
            files={
                "file": (
                    "not_an_image.jpg",
                    b"definitely not image data",
                    "image/jpeg",
                )
            },
        )
        assert response.status_code == 422

    def test_certainty_tier_is_valid_value(
        self,
        api_client: TestClient,
        jpeg_bytes: bytes,
    ) -> None:
        """certainty_tier field is one of the expected tier strings."""
        response = api_client.post(
            "/api/v1/classify",
            files={"file": ("xray.jpg", jpeg_bytes, "image/jpeg")},
        )
        tier = response.json()["certainty_tier"]
        assert tier in {"HIGH", "MODERATE", "LOW"}
