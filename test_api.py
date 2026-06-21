# test_api.py - API endpoint tests
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from main import app

# Context manager triggers lifespan startup (loads the model), matching real server behavior
client = TestClient(app)

def test_health():
    """Health endpoint confirms model is loaded (after startup)."""
    with TestClient(app) as c:   # 'with' triggers the lifespan model-load
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is True
        assert body["num_classes"] == 15


def make_test_image():
    """Create a small valid PNG in memory for testing."""
    img = Image.new('RGB', (512, 512), color='gray')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def test_root():
    """Root endpoint returns API info."""
    r = client.get("/")
    assert r.status_code == 200
    assert "disclaimer" in r.json()




def test_predict_valid_image():
    """A valid image returns a well-formed detection response."""
    img = make_test_image()
    r = client.post("/predict", files={"file": ("test.png", img, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert "detections" in body
    assert "num_detections" in body
    assert body["input_type"] == "image"
    assert isinstance(body["detections"], list)


def test_predict_rejects_bad_extension():
    """A non-image file is rejected with 400."""
    fake = io.BytesIO(b"this is not an image")
    r = client.post("/predict", files={"file": ("test.txt", fake, "text/plain")})
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_predict_rejects_bad_threshold():
    """An out-of-range threshold is rejected with 400."""
    img = make_test_image()
    r = client.post("/predict?score_threshold=5", files={"file": ("test.png", img, "image/png")})
    assert r.status_code == 400
    assert "between 0.0 and 1.0" in r.json()["detail"]


def test_predict_rejects_empty_file():
    """An empty upload is rejected."""
    empty = io.BytesIO(b"")
    r = client.post("/predict", files={"file": ("empty.png", empty, "image/png")})
    assert r.status_code == 400


def test_visualize_returns_image():
    """The visualize endpoint returns a PNG image."""
    img = make_test_image()
    r = client.post("/predict/visualize", files={"file": ("test.png", img, "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"