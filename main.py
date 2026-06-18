# main.py - FastAPI service
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import model as model_module

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dicom", ".dcm"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    model_module.load_model()
    print("Model loaded. API ready.")
    yield
    print("Shutting down.")


app = FastAPI(title="Chest X-Ray Abnormality Detection API",
              description="Educational demo - NOT for clinical use",
              version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Chest X-Ray Detection API",
            "docs": "/docs",
            "disclaimer": "Educational/research demonstration only. Not a diagnostic device."}


@app.get("/health")
def health():
    loaded = model_module._model is not None
    return {"status": "ok" if loaded else "model not loaded",
            "model_loaded": loaded,
            "num_classes": model_module.NUM_CLASSES,
            "device": str(model_module.DEVICE)}


def _load_image_from_upload(contents, ext, content_type):
    """Shared validation + image loading. Returns (image, is_dicom)."""
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413,
            detail=f"File too large. Maximum is {MAX_FILE_SIZE//(1024*1024)} MB.")
    is_dicom = ext in (".dicom", ".dcm") or (content_type == "application/dicom")
    try:
        if is_dicom:
            image = model_module.read_dicom(contents)
        else:
            image = Image.open(io.BytesIO(contents))
            image.verify()
            image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400,
            detail=f"Could not read the file as a valid {'DICOM' if is_dicom else 'image'}. It may be corrupted.")
    return image, is_dicom


def _validate(score_threshold, filename):
    if not (0.0 <= score_threshold <= 1.0):
        raise HTTPException(status_code=400,
            detail=f"score_threshold must be between 0.0 and 1.0, got {score_threshold}.")
    fn = (filename or "").lower()
    if not fn:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = "." + fn.split(".")[-1] if "." in fn else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PNG, JPG, DICOM.")
    return ext


@app.post("/predict")
async def predict(file: UploadFile = File(...), score_threshold: float = 0.25):
    ext = _validate(score_threshold, file.filename)
    contents = await file.read()
    image, is_dicom = _load_image_from_upload(contents, ext, file.content_type)
    try:
        detections = model_module.predict(image, score_threshold=score_threshold, use_tta=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    return {
        "filename": file.filename,
        "input_type": "dicom" if is_dicom else "image",
        "image_size_processed": [model_module.IMG_SIZE, model_module.IMG_SIZE],
        "num_detections": len(detections),
        "detections": detections,
        "disclaimer": "Educational/research demonstration only. Not a diagnostic device."
    }


@app.post("/predict/visualize")
async def predict_visualize(file: UploadFile = File(...), score_threshold: float = 0.25):
    ext = _validate(score_threshold, file.filename)
    contents = await file.read()
    image, is_dicom = _load_image_from_upload(contents, ext, file.content_type)
    detections = model_module.predict(image, score_threshold=score_threshold, use_tta=True)
    annotated = model_module.draw_detections(image, detections)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
        headers={"X-Num-Detections": str(len(detections))})